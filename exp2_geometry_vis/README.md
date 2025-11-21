# 实验2：PCA & 余弦相似度可视化（几何感知）

目标：基于 DINO 特征做 PCA/相似度可视化，展示几何结构；包含论文图的绘制。

代码来源：`./Section2_exp` + 可视化脚本 `./Figure/sec2/*.py`, `./Figure/sup/sup1.py`, `./Section1_vis/umd_overlay.py`（逻辑未改）。

快捷运行
```bash
cd /home/li325/qing_workspace/probing_briding_affordance/experiments/exp2_geometry_vis
# 提取/可视化示例
./run_pca.sh --config ./Section2_exp/config/settings.yaml
```

核心脚本（原路径）：
- PCA/相似度：`Section2_exp/scripts/{cross_domain_pca.py,cross_domain_similarity.py,extract_all.py,knife_roi_pipeline.py,mark_mug_point.py}`
- 模块：`Section2_exp/modules/{feature,roi,pca,similarity,io,config}.py`
- 论文图：`Figure/sec2/sec2_1.py`, `Figure/sec2/sec2_2.py`, `Figure/sup/sup1.py`
- 示例叠加：`Section1_vis/umd_overlay.py`

注意：`cache/`, `data/` 未包含，需要自行准备。
