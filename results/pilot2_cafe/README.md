> 本仓库只包含测量结果。下文提到的视频与图片文件（.mp4、.jpg、.png）含电影画面或生成视频，未上传。

# 两段情节对照实验（2026-10-14）

内容："Two people sit across from each other at a cafe table, talking quietly over coffee. Then they stand up and walk out of the cafe together."
条件：对照组；导演名（" In the style of a <导演> film."）；四系统风格描述（GPT-5.4 撰写，不含导演名，见 code/stereotype_probe2/styles_gpt54.json）。
模型：Wan 3.0（480p，8 秒）与 Veo 3.1 Lite（720p，8 秒），关闭音频，种子 1。26 段，25 段成功；Veo 的 Spielberg 描述条件被平台过滤无输出。费用：描述 0.06 美元，生成 7.30 美元。

文件：compare_cafe_grid.mp4（全部视频同步拼接，左侧为 GPT-5.4 描述）；compare_cafe_grid_1s.jpg 与 _6s.jpg（第 1.5 与第 6 秒画面）；sheet_cafe.jpg（每段 4 帧，红条为检测到的剪切）；metrics.json（剪切与指标）。

观察：
1. 剪辑系统基本没有被执行。GPT-5.4 的描述都要求 2 至 3 个镜头并指定剪切位置，但 11 段描述条件视频中只有 Veo 的 Nolan 出现 1 次剪切；反而 Wan 在只写导演名时，Nolan、Zhang Yimou 出现 2 次剪切、Burton 出现 1 次。
2. 导演名继续触发代表作的图像符号：Veo 的 Kubrick 生成了铺着红色几何花纹地毯的长走廊（与《闪灵》的酒店走廊相似）；Veo 的张艺谋生成了红灯笼、中式花格窗和旗袍；两模型的 Burton 都是哥特尖拱窗。
3. 描述能改变构图与色调，但幅度与方向不稳定：Wes Anderson 描述要求对称与正面，对称度上升（Wan 0.14 至 0.40，Veo 0.04 至 0.23），正面人脸仍为 0；Burton 描述要求低饱和与强反差，Wan 的饱和度反而高于对照组，反差低于对照组。
4. 两个情节（坐着交谈、起身离开）在绝大多数视频中都出现，内容遵循基本正常。
局限：每条件 1 段；指标未校准；剪切检测为直方图法，可能漏检或误检。

更新（同日）：拼接视频拆为两段，并加入真实影片列。compare_cafe_part1.mp4 为对照组与 Nolan、Burton、Kubrick；compare_cafe_part2.mp4 为对照组与 Wes Anderson、张艺谋、Spielberg。真实影片列取自 data/cmd_clips/ref_dialog：只选两张清晰人脸（框高不小于画高 10%，置信度不低于 0.85）出现在多数帧中的镜头，排除每段最后 25 秒（片尾推荐画面），每位导演最多 4 部影片、每部 2 秒。不足 4 部时重复最后一部。只对照“坐着交谈”这一情节，没有“起身离开”的真实镜头。张艺谋的 Hero 一格为士兵与红布灵柩，非对话戏。
