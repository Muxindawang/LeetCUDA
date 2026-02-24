#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
const float exp = 1e-6;

struct Box {
  float x1, y1, x2, y2;
  float conf;
  Box() {}
  Box(float x_1, float y_1, float x_2, float y_2, float conf_) : x1(x_1), y1(y_1), x2(x_2), y2(y_2), conf(conf_) {}
};

float compute_iou(const Box& b1, const Box& b2) {
  float left_x = std::max(b1.x1, b2.x1);
  float right_x = std::min(b1.x2, b2.x2);
  float top_y = std::max(b1.y1, b2.y1);
  float down_y = std::min(b1.y2, b2.y2);

  if (left_x >= right_x || top_y >= down_y) return 0.;
  float inter_area = (right_x - left_x) * (down_y - top_y);
  float area_b1 = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
  float area_b2 = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);

  return inter_area / (area_b1 + area_b2 - inter_area);
}

inline float compute_area(const Box& box) {
  return (box.x2 - box.x1) * (box.y2 - box.y1);
}

inline float compute_inter_area(const Box& b1, const Box& b2) {
  float lx = b1.x1 > b2.x1 ? b1.x1 : b2.x1;
  float rx = b1.x2 > b2.x2 ? b2.x2 : b1.x2;
  float ty = b1.y1 > b2.y1 ? b1.y1 : b2.y1;
  float dy = b1.y2 > b2.y2 ? b2.y2 : b1.y2;

  if (lx >= rx || ty >= dy) return 0.;
  return (rx - lx) * (dy - ty);
}
std::vector<int> nms_cpu(std::vector<Box>& boxes, float iou_threshold) {
  const int n = static_cast<int>(boxes.size());
  if (n == 0) return {};

  // === 1. 预计算所有 area（按原始索引）===
  std::vector<float> areas(n);
  for (int i = 0; i < n; ++i) {
    areas[i] = (boxes[i].x2 - boxes[i].x1) * (boxes[i].y2 - boxes[i].y1);
  }

  // === 2. 按置信度降序排序索引 ===
  std::vector<int> idxs(n);
  std::iota(idxs.begin(), idxs.end(), 0);
  std::sort(idxs.begin(), idxs.end(),
            [&](int a, int b) { return boxes[a].conf > boxes[b].conf; });

  // === 3. NMS 主循环（极致内联）===
  std::vector<bool> suppressed(n, false);
  std::vector<int> keep;
  keep.reserve(n);

  for (int i = 0; i < n; ++i) {
    const int idx_i = idxs[i];
    if (suppressed[idx_i]) continue;
    keep.push_back(idx_i);

    const float x1_i = boxes[idx_i].x1;
    const float y1_i = boxes[idx_i].y1;
    const float x2_i = boxes[idx_i].x2;
    const float y2_i = boxes[idx_i].y2;
    const float area_i = areas[idx_i];

    for (int j = i + 1; j < n; ++j) {
      const int idx_j = idxs[j];
      if (suppressed[idx_j]) continue;

      // 提前计算交集边界（无函数调用）
      const float left = (x1_i > boxes[idx_j].x1) ? x1_i : boxes[idx_j].x1;
      const float right = (x2_i < boxes[idx_j].x2) ? x2_i : boxes[idx_j].x2;
      const float top = (y1_i > boxes[idx_j].y1) ? y1_i : boxes[idx_j].y1;
      const float bottom = (y2_i < boxes[idx_j].y2) ? y2_i : boxes[idx_j].y2;

      if (left >= right || top >= bottom) continue;

      const float inter_area = (right - left) * (bottom - top);
      const float iou = inter_area / (area_i + areas[idx_j] - inter_area);

      if (iou > iou_threshold) {
        suppressed[idx_j] = true;
      }
    }
  }

  return keep;
}

// ====== 测试主函数 ======
int main() {
    // 构造测试 boxes: [x1, y1, x2, y2, conf]
    std::vector<Box> boxes = {
        Box(10, 10, 50, 50, 0.95f),   // box A - 高分
        Box(15, 15, 55, 55, 0.90f),   // box B - 与A高重叠
        Box(100, 100, 150, 150, 0.85f), // box C - 独立
        Box(105, 105, 145, 145, 0.80f), // box D - 与C高重叠
        Box(200, 200, 220, 220, 0.75f)  // box E - 独立小框
    };

    float iou_thresh = 0.5f;
    std::vector<int> keep_indices = nms_cpu(boxes, iou_thresh);

    std::cout << "NMS keep " << keep_indices.size() << " boxes:\n";
    for (int idx_i : keep_indices) {
        const auto& b = boxes[idx_i];
        std::cout << "  Index " << idx_i 
                  << ": [" << b.x1 << ", " << b.y1 << ", " << b.x2 << ", " << b.y2 
                  << "] conf=" << b.conf << "\n";
    }

    // 预期输出：保留索引 0 (A), 2 (C), 4 (E)
    // 因为 B 被 A 抑制，D 被 C 抑制

    return 0;
}


// 优化1 预先计算所有box的area