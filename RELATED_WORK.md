# 相关工作

按与本项目的关系分组。每条只列文献与一句话说明。整理日期：2026-10-14。

## 1. 电影化视频生成的评测

**最接近的工作**
- **EvalVerse**: Pipeline-Aware and Expert-Calibrated Benchmarking for Professional Cinematic Video Generation. Yang et al., arXiv 2605.23271, 2026.
  - 前期、拍摄、后期三阶段，45 个子维度。
  - 从真实影片构造测试题；打分方式为微调 VLM 加人工排序。
  - 没有参数分布比较。
- **FilmBench**: A Film-Grade Benchmark for Cinematic Video Generation. Wang et al., arXiv 2607.24241, 2026.
  - 含“导演风格”主观子项，以及 FilmOps 测量算子。
- **Stable Cinemetrics**: Structured Taxonomy and Evaluation for Professional Video Generation. Chatterjee et al., NeurIPS 2025, arXiv 2509.26555.
  - 76 个控制节点，由从业者打分。
- **CineTechBench**: A Benchmark for Cinematographic Technique Understanding and Generation. Wang et al., NeurIPS 2025 D&B, arXiv 2505.15145.
  - 逐片段比较生成视频与原片的相机轨迹。

**通用与美学评测**
- VBench, VBench++, VBench-2.0. Huang et al., CVPR 2024, arXiv 2311.17982；arXiv 2411.13503；Zheng et al., arXiv 2503.21755。
- AesRM / AesVideo-Bench. Han et al., arXiv 2604.28078, 2026.
- VGA-Bench / VGA-BenchV2. Jiang et al., arXiv 2604.10127；arXiv 2608.25452, 2026.
- EmoTrace-Bench. Wang et al., arXiv 2610.15452, 2026.
  - 发现在提示中加入更多电影语言不一定改善生成效果。
- DynamicEval. Babu et al., arXiv 2510.07441, 2025.
- VGIF-Score. Xu et al., PRCV 2026, arXiv 2607.13527.

**相机控制**
- CameraCtrl. He et al., arXiv 2404.02101, 2024.
- CamI2V. Zheng et al., arXiv 2410.15957, 2024.
- AC3D. Bahmani et al., CVPR 2025, arXiv 2411.18673.
- CamProbe: Probing into Camera Control of Video Models. Hou and Rupprecht, arXiv 2605.14815, 2026.
  - 指出文本诱导失败不等于模型做不到。
- ShotVerse. Yang et al., SIGGRAPH Asia 2026, arXiv 2603.11421.
- PACE: Precise AI Cinematic Expression. Duan et al., arXiv 2609.19853, 2026.

**多镜头与剪辑**
- MSAVBench. Wei et al., arXiv 2605.20183, 2026.
- CutCraft. Zeng et al., arXiv 2609.08275, 2026.
- HoloCine. Meng et al., CVPR 2026, arXiv 2510.20822.
- MuSS. Zhang et al., ACM MM 2026, arXiv 2604.23789.
- PersonaShot. Wang et al., arXiv 2608.16717, 2026.
- DirectorBench. Chen et al., arXiv 2605.30090, 2026.
- CineOrchestra (CineBench). Girish et al., arXiv 2606.13768, 2026.
- DramaChain Bench. Shi et al., arXiv 2609.00646, 2026.
- DramaDirector (DramaBoard). Zhou et al., arXiv 2606.24107, 2026.
- MTAVG-Bench 2.0. Li et al., arXiv 2605.28035, 2026.

## 2. 以参考为条件的风格生成
- StyleMaster: Stylize Your Video with Artistic Generation and Translation. Ye et al., CVPR 2025, arXiv 2412.07744.
  - 以单张参考图做画风化。
- Measuring Style Similarity in Diffusion Models (CSD). Somepalli et al., arXiv 2404.01292, 2024.
- CamCloneMaster. arXiv 2506.03140, 2025.
  - 照一段参考视频复制运镜。
- Follow-Your-Motion. arXiv 2506.05207, 2025.
- JAWS: Just A Wild Shot for Cinematic Transfer in Neural Radiance Fields. Wang et al., CVPR 2023, arXiv 2303.15427.
- Cinematic Behavior Transfer via NeRF-based Differentiable Filming. Jiang, Rao, Wang, Lin, Dai, CVPR 2024.
- E.T. the Exceptional Trajectories: Text-to-camera-trajectory generation with character awareness. Courant et al., ECCV 2024, arXiv 2407.01516.
  - 从 CondensedMovies 还原约 11.5 万段相机轨迹。

## 3. 电影摄影的理解与测量
**理解基准与专用模型**
- ShotBench / ShotVL. Liu et al., arXiv 2506.21356, 2025.
- RefineShot. Wu et al., arXiv 2510.02423, 2025.
- CameraBench: Towards Understanding Camera Motions in Any Video. Lin et al., NeurIPS 2025, arXiv 2504.15376.
- ACaM: Natural Language Camera Movement Understanding. Tan et al., ECCV 2026, arXiv 2607.03043.
- CineCap. Mao et al., arXiv 2606.24636, 2026.
- CHAI: Building a Precise Video Language with Human-AI Oversight. Lin et al., CVPR 2026, arXiv 2604.21718.
- SLU-SUITE. Liu et al., arXiv 2603.18448, 2026.
- CinematicVQA. Xing et al., arXiv 2609.28813, 2026.
- TAKE 85. Shinozaki-Conefrey et al., ECCV 2026 Workshop, arXiv 2608.30068.

**电影数据集**
- MovieNet. Huang et al., ECCV 2020, arXiv 2007.10937.
- MovieShots. Rao et al., ECCV 2020, arXiv 2008.03548.
- AVE: The Anatomy of Video Editing. Argaw et al., ECCV 2022, arXiv 2207.09812.
- CineScale / CineScale2. Savardi et al., Data in Brief 2021, 2023.
- MovieCuts. Pardo et al., CVPR 2022, arXiv 2109.05569.

**相机、焦距、景深、光照与构图的测量工具**
- MegaSaM. Li et al., CVPR 2025, arXiv 2412.04463.
- VGGT. Wang et al., CVPR 2025, arXiv 2503.11651.
- GeoCalib. Veicht et al., ECCV 2024, arXiv 2409.06704.
- AnyCalib. Tirado-Garín and Civera, ICCV 2025, arXiv 2503.12701.
- Depth Pro. Bochkovskii et al., ICLR 2025, arXiv 2410.02073.
- Deep Defocus Map Estimation Using Domain Adaptation. Lee et al., CVPR 2019.
- SfSNet. Sengupta et al., CVPR 2018, arXiv 1712.01261.
- Deep Single-Image Portrait Relighting. Zhou et al., ICCV 2019.
- 6DRepNet. Hempel et al., ICIP 2022, arXiv 2202.12555.
- SAMP-Net: Image Composition Assessment with Saliency-augmented Multi-pattern Pooling. Zhang et al., BMVC 2021, arXiv 2104.03133.
- TransNetV2. Souček and Lokoč, arXiv 2008.04838, 2020.
- AutoShot. Zhu et al., CVPRW 2023, arXiv 2304.06116.
- Auteur: Language-Driven Cinematographic Framing for Human-Centric Video Generation. Kizil et al., NeurIPS 2026, arXiv 2606.01900.

## 4. 色彩与调色
**电影色彩分析**
- VIAN: A Visual Annotation Tool for Film Analysis. Halter et al., Computer Graphics Forum 2019.
- Quicker, faster, darker: Changes in Hollywood film over 75 years. Cutting et al., i-Perception 2011.
- Measuring Colourfulness in Natural Images. Hasler and Süsstrunk, SPIE 2003.
- Color Harmonization. Cohen-Or et al., SIGGRAPH 2006.

**调色估计与色彩迁移**
- Neural Preset for Color Style Transfer. Ke et al., CVPR 2023, arXiv 2303.13511.
- Learning Image-adaptive 3D Lookup Tables. Zeng et al., TPAMI 2020, arXiv 2009.14468.
- Exposure: A White-Box Photo Post-Processing Framework. Hu et al., ACM TOG 2018.
- Adaptive Color Grading. Canham, Punnappurath, Brown, CIC 2026, arXiv 2609.21169.
- Video Color Grading via Look-Up Table Generation. Shin et al., ICCV 2025, arXiv 2508.00548.
- AceTone. Ma et al., CVPR 2026, arXiv 2604.00530.

**光源色与本征分解**
- Auto White-Balance Correction for Mixed-Illuminant Scenes. Afifi et al., WACV 2022, arXiv 2109.08750.
- Colorful Diffuse Intrinsic Image Decomposition in the Wild. Careaga and Aksoy, SIGGRAPH Asia 2024, arXiv 2409.13690.
- Fast Fourier Color Constancy. Barron and Tsai, CVPR 2017.
- DiffusionRenderer. Liang et al., CVPR 2025, arXiv 2501.18590.

**视频生成中的色彩与重打光**
- GenColorBench. Butt et al., CVPR 2026, arXiv 2510.20586.
- Color Conditional Generation with Sliced Wasserstein Guidance. Lobashev et al., NeurIPS 2025, arXiv 2503.19034.
- UniLumos. Liu et al., NeurIPS 2025, arXiv 2511.01678.
- Light-A-Video. Zhou et al., arXiv 2502.08590, 2025.
- LumiSculpt. Zhang et al., arXiv 2410.22979, 2024.
- LiVER. Cai et al., CVPR 2026, arXiv 2604.07966.

## 5. 质感：颗粒、锐度、光晕与运动模糊
- Film Grain Synthesis for AV1 Video Codec. Norkin and Birkbeck, DCC 2018.
- FGA-NN: Film Grain Analysis Neural Network. Ameur et al., arXiv 2506.14350, 2025.
- Film Grain Rendering and Parameter Estimation. Zhang et al., ACM TOG 2023.
- Nonparametric Multiscale Blind Estimation of Intensity-Frequency-Dependent Noise. Colom et al., IEEE TIP 2015.
- Single-Image Noise Level Estimation for Blind Denoising. Liu et al., IEEE TIP 2013.
- A No-Reference Perceptual Blur Metric. Marziliano et al., ICIP 2002.
- CPBD: A No-Reference Image Blur Metric. Narvekar and Karam, IEEE TIP 2011.
- Video Shutter Angle Estimation Using Optical Flow and Linear Blur. Korčák and Matas, CVWW 2024, arXiv 2303.10247.
- ProMist-5K. Lei et al., ICASSP 2026, arXiv 2601.19295.
- Deep Analog: Open-Set Film Emulation with Reference-Conditioned 3D LUTs. Mu, arXiv 2608.14702, 2026.

## 6. 数据来源与基准方法论
- Condensed Movies: Story Based Retrieval with Contextual Embeddings. Bain et al., ACCV 2020, arXiv 2005.04208.
- Who is the director of this movie? Automatic style recognition based on shot features. Svanera et al., IEEE MultiMedia 2019, arXiv 1807.09560.
- BetterBench: Assessing AI Benchmarks, Uncovering Issues, and Establishing Best Practices. Reuel et al., NeurIPS 2024 D&B, arXiv 2411.12990.
