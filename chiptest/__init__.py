"""芯片测试（ATE 日志）数据分析：面向芯片测试工程岗位的第二个方向。

与 aoi/ 并列，复用同一套"口径先行、对照分析、诚实说明局限"的做法：
  logparse   流式解析 STDF 风格文本日志（海量日志、脏数据、单位混用）
  data       质量体检、清洗、派生字段
  yield_analysis / attribution   良率、复测、"硬件还是硅"的归因
  parametric  Cpk、稳健离群值、PAT
  wafer      晶圆图：径向良率、划伤集群
  efficiency  测试项冗余、测试顺序优化（给 DFT/测试方案的反馈）
  evaluation  仅模拟数据可得的真值评估（过杀/漏测、PAT 的召回）
"""
