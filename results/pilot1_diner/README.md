> 本仓库只包含测量结果。下文提到的视频与图片文件（.mp4、.jpg、.png）含电影画面或生成视频，未上传。

# 导演名提示小实验（2026-10-14）

内容固定为 "Two people sit across from each other at a diner table, talking quietly over coffee."，句末追加 ", in the style of a <导演> film"；对照组不加。种子 1、2。Wan 3.0 为 720p 5 秒，Veo 3.1 Lite 为 720p 4 秒，均关闭音频，经 OpenRouter 调用。共 28 段，成功 27 段：Wan 的 Spielberg 种子 2 被平台拦截，理由为 "The output content is suspected to include real human faces."。花费约 7.2 美元。

文件：sheet_*.jpg 为对照图（每行一种风格，每个种子取首、中、尾三帧）；stats.json 为逐段指标；脚本在 code/stereotype_probe/。

定性观察：
1. 两个模型的 28 段几乎全是同一种拍法：侧面中景双人镜头、机位基本不动、无剪切。导演名改变的主要是画面内容（服装、化妆、人物族裔、场景陈设），而不是景别、视角、运镜、构图。
2. 刻板化明显：Burton 两个模型都把人物画成哥特妆容；Wan 把张艺谋理解为中式茶馆与中国演员，Veo 只给人物换了红衣服；Wes Anderson 两个模型都换成粉彩服装与薄荷绿陈设，但没有正面对称构图。
3. 部分导演几乎无效：Veo 的 Kubrick 与对照组人物、服装、构图几乎相同，没有单点透视与广角。
4. 色彩层有响应：Veo 的 Nolan 饱和度 0.17，对照组 0.41；Wan 的 Burton 亮度 15，对照组 35。

局限：每组只有 2 段、只有 1 个内容；指标为简单像素统计，左右对称度受画面明暗影响（Wan Burton 最暗，对称度最高）。

## 合并对比（compare.py 生成）
- compare_grid.mp4：7 行风格 × 4 列（Wan 种子 1、2，Veo 种子 1、2）同步播放 4 秒；compare_grid_mid.jpg 为中间帧。
- compare_metrics.png：6 项指标的分组柱状图，柱为 2 段均值，黑点为每段，虚线为无风格对照。
- compare_table.csv：同上数值。

合并后新增的观察：
5. Veo 的两个种子几乎生成同一画面（同样的演员、服装、构图），而且 Kubrick、张艺谋与对照组共用同一位短发女演员；Wan 的两个种子之间差异较大。种子间多样性低是另一种刻板化。
6. 两个模型对同一导演的反应方向不一致：Kubrick 的饱和度 Wan 升到 0.47、Veo 降到 0.27；Nolan 的饱和度 Wan 0.42、Veo 0.17。说明“导演名对应什么参数”在模型之间并无共识。

## 与真实片段对比（refs.py 生成，2026-10-14）
真实片段：CMD 中每位导演 3 至 4 部影片、每部 2 段（优先对话戏），共 46 段（Burton 的 Pee-wee 两段已下架）；视频存于 data/cmd_clips/ref6/，只用于测量，不分发。处理：裁黑边，去掉开头 2 秒与 MovieClips 片尾推荐画面，约每秒 5 帧。
- real_vs_generated.jpg：左为真实片段中最长两个镜头的中间帧，右为生成结果。
- real_vs_generated_metrics.png：真实（绿，每点一段）与两个模型的 6 项指标。运动量因帧率与剪切不同不可比，已换成红绿 a*。
- data/cmd_clips/ref6/ref_stats.json：逐段数值，含镜头数与平均镜头长度。

已知问题：Kubrick 选中的多为黑白片（Dr. Strangelove、Killer's Kiss、Paths of Glory），饱和度 0.11，且基于色彩直方图的切镜头在黑白片上失效（平均镜头长度 90 秒不可信）；Jaws 一段仍残留旧版片尾画面；张艺谋三部片中《长城》多为 CG 镜头。

## 实验 1 与 2（2026-10-14）
- 实验 1（verbalize.py → verbalized.json）：Qwen 3.8 Max、Gemini 3.8 Flash、GPT-5.4 把每位导演在这场戏中的拍法说成参数。三者都说 5 秒只有 1 个镜头，景别中景或中近景，机位多为过肩或侧面；只有 Kubrick、Wes Anderson 被说成中心对称。费用 0.18 美元。
- 实验 2（--style_key styles_traits）：不写导演名，只写我自行扩写的英文特征（见 prompts_api.json 的 styles_traits，已作废）。Veo 种子 1、2，Wan 种子 1，共 18 段，约 4.0 美元。
- final_compare.mp4 / .jpg：真实影片（4 部片轮播）| 导演名 Wan、Veo | 特征描述 Wan、Veo，左侧附 GPT-5.4 说出的拍法。

观察：导演名主要触发人物与服装（Burton 的哥特妆容、张艺谋的中式茶馆），特征描述则改变光线与色彩（Spielberg 出现暖色逆光与光束，张艺谋出现大面积红黄色块，Wan 的 Kubrick 出现纵深单点透视）。但特征描述不一定更接近真实片段的测量值：张艺谋特征描述的饱和度 0.71 至 0.73、偏黄 42，真实片段 0.28 与 1.4；Wes Anderson 写“粉彩”后偏黄从 11 降到 4 至 6，真实片段为 17。影评式描述本身也是刻板印象，目标应来自真实片段的测量。

## 特征表对齐版（2026-10-14 下午，现行版本）
- 特征表来源：code/stereotype_probe/traits.json 由我（Claude）依据常见影评说法凭印象整理，未经文献核对，也未经测量；真实对话镜头只部分支持它（见 trait_comparison.md）。
- 生成：prompts_api.json 的 styles_table，为 traits.json 逐条直译，不增删；Wan 与 Veo 各种子 1、2，共 24 段（文件名含 table_），6.54 美元。早先的 traits_ 版本由我自行扩写，作废，仅留档。
- 原片：data/cmd_clips/ref_dialog/，每位导演最多 16 段非动作片段（张艺谋放宽），按镜头筛对话镜头（YuNet 检测到 1 至 3 张脸的帧占 60% 以上，且镜头至少 1 秒），共 1,122 个对话镜头。
- 指标：code/stereotype_probe/trait_metrics.py，特征表每一条对应一个指标，测不了的列为人评。
- 输出：trait_comparison.md / .csv（英文对比表），final_compare_en.mp4 / .jpg（英文对比视频）。
- 已知问题：指标未经人工校准（正面朝向用眼鼻关键点近似，横移指标对手持抖动敏感，灭点检测粗糙）；对比视频真实列取“最长对话镜头”，偶有电视画面等非典型镜头；CMD 覆盖的影片有限（Wes Anderson 只有早期三部，张艺谋只有《英雄》《十面埋伏》《长城》）。
