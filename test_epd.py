"""
EPD v5 核心逻辑测试 - 简化版本
直接验证 EPD 的核心计算公式，不依赖完整函数
"""
import torch


def compute_epd_weights(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    epd_lambda: float = 0.8,
    epd_tau: float = 0.5
):
    """
    EPD 核心计算逻辑（从 apply_teacher_entropy_weighting 的 v5_epd 分支提取）
    """
    with torch.no_grad():
        # 计算 student 熵
        student_probs = student_topk_log_probs.exp()
        safe_student_logp = torch.clamp(student_topk_log_probs, min=-100.0)
        student_entropy = -(student_probs * safe_student_logp).sum(dim=-1)  # (B, T)

        # 计算 teacher 熵
        teacher_probs = teacher_topk_log_probs.exp()
        safe_teacher_logp = torch.clamp(teacher_topk_log_probs, min=-100.0)
        teacher_entropy = -(teacher_probs * safe_teacher_logp).sum(dim=-1)  # (B, T)

        # 坍缩风险：teacher 比 student 确定多少
        collapse_risk = torch.clamp(student_entropy - teacher_entropy, min=0)  # ≥0
        normalized_collapse = collapse_risk / (student_entropy + 1e-8)
        normalized_collapse = torch.clamp(normalized_collapse, max=1.0)  # ∈ [0, 1]

        # Sigmoid 调制
        sigmoid_weights = 1.0 - epd_lambda * torch.sigmoid(normalized_collapse / epd_tau)

        # 关键修复：当 collapse_risk = 0 时（teacher 不比 student 更确定），完全不保护
        confidence_weights = torch.where(
            collapse_risk > 0,
            sigmoid_weights,
            torch.ones_like(sigmoid_weights)  # w_distill = 1，完全不保护
        )

        # 应用 mask
        confidence_weights = confidence_weights.masked_fill(loss_mask == 0, 0.0)

        return confidence_weights, {
            'student_entropy': student_entropy,
            'teacher_entropy': teacher_entropy,
            'collapse_risk': collapse_risk,
            'normalized_collapse': normalized_collapse
        }


def test_epd_basic():
    """测试 1: EPD 基本功能"""
    print("=" * 60)
    print("测试 1: EPD 基本功能")
    print("=" * 60)

    # 创建 mock 数据
    batch_size = 2
    seq_len = 10
    topk = 50

    # Student log probs (高熵 - 不确定，接近均匀分布)
    student_topk_log_probs = torch.log_softmax(
        torch.randn(batch_size, seq_len, topk) * 0.1,  # 小方差 -> 接近均匀 -> 高熵
        dim=-1
    )

    # Teacher log probs (低熵 - 确定，集中在少数 token)
    teacher_logits = torch.randn(batch_size, seq_len, topk)
    teacher_logits[..., 0] += 10.0  # 让第一个 token 概率很高
    teacher_topk_log_probs = torch.log_softmax(teacher_logits, dim=-1)

    # Loss mask
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[0, 8:] = 0
    loss_mask[1, 7:] = 0

    # 计算权重
    weights, metrics = compute_epd_weights(
        student_topk_log_probs,
        teacher_topk_log_probs,
        loss_mask,
        epd_lambda=0.8,
        epd_tau=0.5
    )

    print(f"输入形状: {student_topk_log_probs.shape}")
    print(f"Student 熵均值: {metrics['student_entropy'][loss_mask == 1].mean():.4f}")
    print(f"Teacher 熵均值: {metrics['teacher_entropy'][loss_mask == 1].mean():.4f}")
    print(f"Collapse risk 均值: {metrics['collapse_risk'][loss_mask == 1].mean():.4f}")
    print(f"权重范围: [{weights[loss_mask == 1].min():.4f}, {weights[loss_mask == 1].max():.4f}]")
    print(f"权重均值: {weights[loss_mask == 1].mean():.4f}")

    # 验证
    assert weights.shape == (batch_size, seq_len), "权重形状不匹配"
    assert not torch.isnan(weights).any(), "权重包含 NaN"
    assert (weights[loss_mask == 1] >= 0.2).all(), "权重应该 >= 1-λ = 0.2"
    assert (weights[loss_mask == 1] <= 1.0).all(), "权重应该 <= 1.0"
    assert (weights[loss_mask == 0] == 0.0).all(), "padding 位置的权重应该为 0"
    assert metrics['collapse_risk'][loss_mask == 1].mean() > 0, "Collapse risk 应该大于 0"

    print("\n✓ 测试 1 通过")
    return True


def test_epd_self_regulation():
    """测试 2: EPD 自调节机制"""
    print("\n" + "=" * 60)
    print("测试 2: EPD 自调节机制")
    print("=" * 60)

    batch_size = 1
    seq_len = 5
    topk = 50
    epd_lambda = 0.8
    epd_tau = 0.5

    # Teacher 始终很确定 (低熵)
    teacher_logits = torch.randn(batch_size, seq_len, topk)
    teacher_logits[..., 0] += 10.0  # 集中在第一个 token
    teacher_certain = torch.log_softmax(teacher_logits, dim=-1)

    # 场景 1: Student 高熵 (训练初期 - 接近均匀分布)
    student_high_entropy = torch.log_softmax(
        torch.randn(batch_size, seq_len, topk) * 0.1,  # 小方差 -> 接近均匀 -> 高熵
        dim=-1
    )

    mask = torch.ones(batch_size, seq_len)
    weights1, metrics1 = compute_epd_weights(
        student_high_entropy, teacher_certain, mask,
        epd_lambda, epd_tau
    )

    # 场景 2: Student 低熵 (训练后期 - 也很确定)
    student_logits = torch.randn(batch_size, seq_len, topk)
    student_logits[..., 0] += 8.0  # 也比较集中，但不如 teacher
    student_low_entropy = torch.log_softmax(student_logits, dim=-1)

    weights2, metrics2 = compute_epd_weights(
        student_low_entropy, teacher_certain, mask,
        epd_lambda, epd_tau
    )

    print(f"场景 1 (Student 高熵 - 训练初期):")
    print(f"  Student 熵: {metrics1['student_entropy'].mean():.4f}")
    print(f"  Collapse risk: {metrics1['collapse_risk'].mean():.4f}")
    print(f"  权重均值: {weights1.mean():.4f}")

    print(f"\n场景 2 (Student 低熵 - 训练后期):")
    print(f"  Student 熵: {metrics2['student_entropy'].mean():.4f}")
    print(f"  Collapse risk: {metrics2['collapse_risk'].mean():.4f}")
    print(f"  权重均值: {weights2.mean():.4f}")

    # 验证自调节：高熵时保护更强（权重更低）
    assert weights1.mean() < weights2.mean(), \
        f"高熵时权重应更低 (更强保护): {weights1.mean():.4f} vs {weights2.mean():.4f}"

    print("\n✓ 测试 2 通过：自调节机制正确")
    return True


def test_epd_edge_cases():
    """测试 3: 边界情况"""
    print("\n" + "=" * 60)
    print("测试 3: 边界情况")
    print("=" * 60)

    batch_size = 1
    seq_len = 3
    topk = 50

    # 场景 A: Student 和 Teacher 熵相同 (不需要保护)
    same_entropy = torch.log_softmax(
        torch.randn(batch_size, seq_len, topk),
        dim=-1
    )

    mask = torch.ones(batch_size, seq_len)
    weights_a, metrics_a = compute_epd_weights(
        same_entropy, same_entropy, mask,
        epd_lambda=0.8, epd_tau=0.5
    )

    print(f"场景 A (Student = Teacher):")
    print(f"  Collapse risk: {metrics_a['collapse_risk'].mean():.6f}")
    print(f"  权重均值: {weights_a.mean():.4f}")

    # 当 student 和 teacher 熵相同时，collapse_risk = 0，权重 = 1（完全不保护）
    assert torch.allclose(weights_a, torch.ones_like(weights_a), atol=0.01), \
        "当熵相同时，权重应接近 1.0（不保护）"

    # 场景 B: Student 熵 < Teacher 熵 (反向情况，不需要保护)
    # 要让 student 熵更低（更确定的分布），teacher 熵更高（更均匀的分布）
    student_low = torch.log_softmax(
        torch.randn(batch_size, seq_len, topk) * 3,  # 大系数 -> 分布更尖锐 -> 低熵
        dim=-1
    )
    teacher_high = torch.log_softmax(
        torch.randn(batch_size, seq_len, topk) * 0.3,  # 小系数 -> 接近均匀分布 -> 高熵
        dim=-1
    )

    weights_b, metrics_b = compute_epd_weights(
        student_low, teacher_high, mask,
        epd_lambda=0.8, epd_tau=0.5
    )

    print(f"\n场景 B (Student < Teacher):")
    print(f"  Student 熵: {metrics_b['student_entropy'].mean():.4f}")
    print(f"  Teacher 熵: {metrics_b['teacher_entropy'].mean():.4f}")
    print(f"  Collapse risk: {metrics_b['collapse_risk'].mean():.6f}")
    print(f"  权重均值: {weights_b.mean():.4f}")

    # 当 student 熵 < teacher 熵时，collapse_risk = 0，权重 = 1-λ*sigmoid(0) = 1-λ*0.5
    assert metrics_b['collapse_risk'].mean() < 0.01, "反向情况时 collapse_risk 应接近 0"

    print("\n✓ 测试 3 通过：边界情况处理正确")
    return True


def test_epd_formula():
    """测试 4: 验证公式正确性"""
    print("\n" + "=" * 60)
    print("测试 4: 验证 EPD 公式")
    print("=" * 60)

    # 手动计算一个简单例子
    batch_size = 1
    seq_len = 1
    topk = 10

    # Student: 均匀分布 (高熵)
    student_uniform = torch.log_softmax(
        torch.zeros(batch_size, seq_len, topk),
        dim=-1
    )  # 每个 token 概率 = 0.1

    # Teacher: 集中在一个 token (低熵)
    teacher_peaked = torch.full((batch_size, seq_len, topk), -10.0)
    teacher_peaked[0, 0, 0] = 0.0
    teacher_peaked = torch.log_softmax(teacher_peaked, dim=-1)

    mask = torch.ones(batch_size, seq_len)
    weights, metrics = compute_epd_weights(
        student_uniform, teacher_peaked, mask,
        epd_lambda=0.8, epd_tau=0.5
    )

    # 手动验证
    h_student = -10 * (0.1 * torch.log(torch.tensor(0.1)))  # 均匀分布的熵
    h_teacher_approx = -torch.log(torch.tensor(0.99))  # 接近 0

    print(f"Student 熵 (理论值): {h_student.item():.4f}")
    print(f"Student 熵 (计算值): {metrics['student_entropy'].item():.4f}")
    print(f"Teacher 熵 (近似): {h_teacher_approx.item():.4f}")
    print(f"Teacher 熵 (计算值): {metrics['teacher_entropy'].item():.4f}")
    print(f"Collapse risk: {metrics['collapse_risk'].item():.4f}")
    print(f"Normalized collapse: {metrics['normalized_collapse'].item():.4f}")

    # 手动计算期望权重
    normalized = min((h_student.item() - h_teacher_approx.item()) / h_student.item(), 1.0)
    expected_weight = 1.0 - 0.8 * torch.sigmoid(torch.tensor(normalized / 0.5)).item()

    print(f"\n期望权重 (手动计算): {expected_weight:.4f}")
    print(f"实际权重: {weights.item():.4f}")

    assert abs(weights.item() - expected_weight) < 0.01, \
        f"权重计算错误: {weights.item():.4f} vs {expected_weight:.4f}"

    print("\n✓ 测试 4 通过：公式实现正确")
    return True


if __name__ == "__main__":
    try:
        test_epd_basic()
        test_epd_self_regulation()
        test_epd_edge_cases()
        test_epd_formula()

        print("\n" + "=" * 60)
        print("✓✓✓ 所有测试通过 ✓✓✓")
        print("=" * 60)
        print("\nEPD v5 核心逻辑验证完成！")
        print("下一步：在真实训练环境中集成测试")

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
