# Datasets for BEV Perception Overview

This branch is dedicated to building the dataset module for the SafeBEV project, mainly including:

### Jump to:

- [Single-Modality Vehicle-Side Perceptionn Datasets](#Single-Modality-anchor)
- [Multi-Modality Vehicle-Side Perception Datasets](#Multi-Modality-anchor)
- [V2V Datasets](#v2v-datasets-anchor)
- [V2I Datasets](#v2i-datasets-anchor)
- [V2X Datasets](#v2x-datasets-anchor)
- [I2I Datasets](#i2i-datasets-anchor)
- [Roadside Datasets](#roadside-datasets-anchor)

## Multi-Modality Vehicle-Side Perception Datasets <a id="Multi-Modality-anchor"></a>

| Acronym               | Paper Title                                                                                                          | Venue           | Link                                                                                          |
|-----------------------|----------------------------------------------------------------------------------------------------------------------|-----------------|-----------------------------------------------------------------------------------------------|
| **Argoverse2**        | Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting                                    | arXiv (2023)    | https://github.com/argoverse/argoverse-api                                                             |
| **ZOD**               | Zenseact Open Dataset: A large-scale and diverse multimodal dataset for autonomous driving                          | ICCV (2023)     | https://zod.zenseact.com/                                                                    |
| **Lyft L5**           | Large Car-following Data Based on Lyft Level-5 Open Dataset                                                          | arXiv (2023)    | https://github.com/RomainLITUD/Car-Following-Dataset-HV-vs-AV                                                             |
| **ONCE-3DLanes**      | ONCE-3DLanes: Building Monocular 3D Lane Detection                                                                  | CVPR (2022)     | https://once-3dlanes.github.io/                                                                                        |
| **OpenLane**          | PersFormer: 3D Lane Detection via Perspective Transformer                                                           | ECCV (2022)     | https://github.com/OpenDriveLab/OpenLane                                                      |
| **SHIFT**             | Shift: a synthetic driving dataset for continuous multi-modal benchmarking                                          | CVPR (2022)     | http://www.vis.xyz/shift                                             |
| **AIODrive**          | All-In-One Drive: A Comprehensive Perception Dataset for Autonomous Driving                                         | NeurIPS (2021)     | http://www.aiodrive.org/                                                     |
| **ONCE**              | One Million Scenes for Autonomous Driving                                                                           | arxiv (2021)     | https://once-for-auto-driving.github.io/                                                      |
| **RADIATE**           | Radiate: A Radar Dataset for Automotive Perception                                                                  | ICRA (2021)     | https://pro.hw.ac.uk/radiate/                                                                 |
| **nuPlan**            | nuPlan: A closed-loop ML-based planning benchmark for autonomous driving                                              | arXiv (2021)    | https://www.nuplan.org/                                                                       |
| **Waymo Open**        | Scalability in Perception for Autonomous Driving: Waymo Open Dataset                                                | CVPR (2020)     | https://waymo.com/open/                                                                       |
| **A2D2**              | A2D2: Audi Autonomous Driving Dataset                                                                                | arxiv (2020)               | https://www.a2d2.audi/a2d2/en.html                                                            |
| **PandaSet**          | PandaSet: Advanced Sensor Suite Dataset for Autonomous Driving                                                      | arXiv (2021)    | https://pandaset.org/                                                                         |
| **KITTI-360**         | KITTI-360: A Novel Dataset and Benchmarks for Urban Scene Understanding                                             | arXiv (2020)    | http://www.cvlibs.net/datasets/kitti-360                                                                                        |
| **STF**               | Scalability in Perception for Autonomous Driving (sub-benchmark of Waymo Open Dataset)                               | CVPR (2020)     | https://www.nuscenes.org/fog                                                                                        |
| **Oxford RadarCar**   | The Oxford Radar RobotCar Dataset: A Radar Extension to the RobotCar Dataset                                         | arxiv (2020)     | https://ori.ox.ac.uk/datasets/radar-robotcar-dataset                                 |
| **CiRUS**             | CiRUS: A Radar-Based Perception Dataset for Autonomous Driving                                                      | arXiv (2020)    | https://www.cirrus-dataset.net/                                                               |
| **VirtualKITTI2**     | Virtual KITTI2                                                                                                        | CVPR (2020)    | https://arxiv.org/abs/2001.10773                                                              |
| **BDD100K**           | BDD100K: A Diverse Driving Video Dataset                                                                             | ECCV (2018)     | https://bdd-data.berkeley.edu/                                                                |
| **nuScenes**          | nuScenes: A Multimodal Dataset for Autonomous Driving                                                               | CVPR (2019)     | https://github.com/nutonomy/nuscenes-devkit                                                                     |
| **H3D**               | H3D: A Multi-Camera and 3D LiDAR Dataset for Autonomous Driving                                                     | ICRA (2019)     | https://usa.honda-ri.com/h3d                                                                  |
| **SemanticKITTI**     | SemanticKITTI: A Large-Scale LiDAR Sequence Dataset for Semantic Segmentation                                        | ICCV (2019)     | http://www.semantic-kitti.org/                                                                |
| **Astyx**             | Astyx HiRes2019: A High-Resolution Radar Dataset                                                                    | arXiv (2019)    |  https://www.cruisemunich.de/                                                              |
| **Argoverse**         | Argoverse: 3D Tracking and Forecasting Dataset                                                                     | CVPR (2019)     | https://www.argoverse.org/                                                                    |
| **A\*3D**             | A\*3D Dataset: Towards Autonomous Driving in Challenging Environments                                               | arXiv (2019)    | https://github.com/I2RDL2/ASTAR-3D                                                            |
| **KAIST**             | KAIST Multi-Spectral Day/Night Dataset for Autonomous Driving                                                       | ICRA (2018)     | http://multispectral.kaist.ac.kr/                                                             |
| **ApolloScape**       | ApolloScape: A Large-Scale Dataset for Autonomous Driving                                                          | CVPR (2018)     | http://apolloscape.auto/                                                                      |
| **Oxford RobotCar**   | 1 year, 1000 km: The oxford robotcar dataset                                                                     | SAGE (2017)     | https://robotcar-dataset.robots.ox.ac.uk/                                                     |
| **Cityscapes**        | Cityscapes: A Dataset for Semantic Urban Scene Understanding                                                        | CVPR (2016)     | https://www.cityscapes-dataset.com/                                                           |
| **SYNTHIA**           | SYNTHIA: A Large Collection of Synthetic Images for Semantic Segmentation                                          | CVPR (2016)     | https://synthia-dataset.net/                                                                  |
| **KITTI**             | KITTI Vision Benchmark Suite                                                                                       | CVPR (2012)     | http://www.cvlibs.net/datasets/kitti/                                                         |

---

## Single-Modality Vehicle-Side Perception Datasets <a id="Single-Modality-anchor"></a>

| Acronym             | Paper Title                                                                            | Venue           | Link                                                                    |
|---------------------|----------------------------------------------------------------------------------------|-----------------|-------------------------------------------------------------------------|
| **ACDC**            | ACDC: The adverse conditions dataset with correspondences for semantic segmentation     | ICCV (2021)     | https://acdc.vision.ee.ethz.ch/                                         |
| **VirtualKITT**     | Virtual KITTI: A Photo-Realistic Synthetic Dataset for Autonomous Driving               | CVPR (2016)     | https://europe.xerox.com/innovation/computer-vision/virtual-kitti       |
| **KITTI MOTS**      | KITTI MOTS: Multi-Object Tracking and Segmentation                                     | arXiv (2019)    | https://arxiv.org/abs/1905.04854                                         |
| **IDD**             | IDD: A Dataset for Exploring Problems of Autonomous Navigation in Unstructured Scenes   | CVPR (2019)     | https://idd.insaan.iiit.ac.in/                                          |
| **DET**             | DET: A High-Resolution DVS Dataset for Lane Extraction in the Wild                      | CVPR (2019)     | https://spritea.github.io/DET/                                                                   |
| **Spatial CNN**     | Spatial as Deep: Spatial CNN for Traffic Scene Understanding                            | AAAI (2019)     | https://ojs.aaai.org/index.php/AAAI/article/view/12301                   |
| **VPGNet**          | VPGNet: Vanishing Point Guided Network for Lane and Road Marking Detection              | CVPR (2017)     | https://github.com/SeokjuLee/VPGNet                                                                  |
| **CamVid**          | CamVid: A High-Definition Ground Truth Video Database                                   | PAMI (2013)     | https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/             |
| **SynFog**          | SynFog: A Photo-Realistic Synthetic Fog Dataset                                         | CVPR (2024)     | _N/A_                                                                   |
| **CAOS**            | CAOS: Scaling Out-of-Distribution Detection for Real-World Autonomous Driving           | arXiv (2019)    | https://arxiv.org/abs/1911.11132                                       |
| **Foggy Cityscapes**| Semantic Foggy Scene Understanding with Synthetic Data               | IJCV (2018)     | https://arxiv.org/pdf/1708.07819                    |
| **Lane Det**| Spatial as Deep: Spatial CNN for Traffic Scene Understanding                | AAAI (2018)     | https://ojs.aaai.org/index.php/AAAI/article/view/12301                    |
| **StreetHazards**| Scaling out-of-distribution detection for real-world settings                | NeurIPS (2022)     | https://github.com/hendrycks/anomaly-seg                     |
---





## Roadside Datasets  <a id="roadside-datasets-anchor"></a>

| Acronym         | Paper Title                                                                                | Venue (Year)    | Link                                                                                       |
|-----------------|--------------------------------------------------------------------------------------------|-----------------|--------------------------------------------------------------------------------------------|
| **Ko-PER**      | Ko-PER: Korea-Perception Infrastructure Dataset                                             | ITSC (2014)     | [[paper](https://ieeexplore.ieee.org/abstract/document/6957976)] [~~code~~] [[project](https://www.uni-ulm.de/in/mrm/forschung/datensaetze.html)]
| **CityFlow**    | CityFlow: A Multi-Camera Multi-Target Tracking Benchmark                                    | CVPR (2019)     | [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_CityFlow_A_City-Scale_Benchmark_for_Multi-Target_Multi-Camera_Vehicle_Tracking_and_CVPR_2019_paper.pdf)] [[code](https://github.com/cityflow-project/CityFlow/)] [[project](https://cityflow-project.github.io/)]  [[doc](https://cityflow.readthedocs.io/en/latest/)]                                                       |
| **INTERACTION** | INTERACTION: A Real-World Multi-Agent Motion Dataset                                        | IROS (2019)     | [[paper](https://arxiv.org/abs/1910.03088)] [~~code~~] [project](https://interaction-dataset.com/)                                                          |
| **CoopInf**     | Cooperative perception for 3D object detection in driving scenarios using infrastructure sensors    | T-ITS (2020)    | [[paper](https://arxiv.org/pdf/1912.12147)] [[~~code~~]()] [[project](https://github.com/eduardohenriquearnold/coop-3dod-infra)]                                   |
| **A9-Dataset**  | A9-Dataset: Provider-Based Cooperative Perception Data                                      | IV (2022)       | [[paper](https://ieeexplore.ieee.org/abstract/document/9827401)] [~~code~~] [[code](https://github.com/providentia-project/a9-dev-kit)] [[project]([https://a9-dataset.com](https://innovation-mobility.com/en/project-providentia/a9-dataset/))]                         |
| **IPS300+**     | IPS300+: Infrastructure Perception Sensor Dataset                                          | ICRA (2022)     | [[paper](https://ieeexplore.ieee.org/abstract/document/9811699)]  [~~code~~]  [[project](http://www.openmpd.com/column/IPS300)]                                                      |
| **Rope3D**      | Rope3D: Infrastructure Point-Cloud Video Segmentation Dataset                               | CVPR (2022)     | [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Ye_Rope3D_The_Roadside_Perception_Dataset_for_Autonomous_Driving_and_Monocular_CVPR_2022_paper.pdf)] [[code]()] [[project](https://thudair.baai.ac.cn/rope)] [~~project~~]                                                           |
| **LUMPI**       | LUMPI: LiDAR-UWB Multi-Point Infrastructure Dataset                                        | IV (2022)       | [[paper](https://ieeexplore.ieee.org/abstract/document/9827157)]  [~~code~~]  [[project](https://data.uni-hannover.de/cs_CZ/dataset/lumpi)]                                           |
| **TUMTraf-I**   | TUMTraf-I: Infrastructure-Only Traffic Dataset                                              | ITSC (2023)     | [[paper](https://ieeexplore.ieee.org/abstract/document/10422289)]  [~~code~~]  [[project](https://innovation-mobility.com/en/project-providentia/a9-dataset/)]                         |
| **RoScenes**    | RoScenes: A Real-World Infrastructure Camera Dataset for 3D Perception                      | ECCV (2024)     | [[paper](https://link.springer.com/chapter/10.1007/978-3-031-72940-9_19)] [[paper](https://arxiv.org/abs/2405.09883)] [[code](https://github.com/roscenes/RoScenes)] [[project](https://roscenes.github.io./})]                                                               |
| **H-V2X**       | H-V2X: Hybrid Camera–Radar Infrastructure Dataset                                           | ECCV (2024)     | [[paper](https://eccv2024.ecva.net/virtual/2024/poster/126)] [~~code~~] [[project](https://pan.quark.cn/s/86d19da10d18)]                                                       |
---


## V2V Datasets <a id="v2v-datasets-anchor"></a>
- **V2V Datasets**: Vehicle-to-vehicle datasets capture collaboration between vehicles, facilitating research on cooperative perception under occlusion, sparse observations, or dynamic driving scenarios.


| Acronym      | Paper Title                                                                         | Venue (Year)       | Link                                                                         |
|--------------|-------------------------------------------------------------------------------------|--------------------|------------------------------------------------------------------------------|
| **COMAP**    | COMAP: A Synthetic Dataset for Collective Multi-Agent Perception of Autonomous Driving | ISPRS Archives (2021) |  [[paper](https://isprs-archives.copernicus.org/articles/XLIII-B2-2021/255/2021/isprs-archives-XLIII-B2-2021-255-2021.pdf)] [[~~code~~]()] [[project](https://demuc.de/colmap/ )]                                                   |
| **CODD**     | Fast and Robust Registration of Partially Overlapping Point Clouds  | IEEE RA-L (2021)   | [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9662220)] [[code](https://github.com/eduardohenriquearnold/fastreg)] [[project](https://github.com/eduardohenriquearnold/fastreg)]                             |
| **OPV2V**    | OPV2V: A Large‐Scale Benchmark for Multi‐View 3D Object Detection under V2V Comm.     | ICRA (2022)        | [[paper](https://arxiv.org/abs/2109.07644)] [[code](https://github.com/DerrickXuNu/OpenCOOD)] [[project](https://mobility-lab.seas.ucla.edu/opv2v)]                                   |
| **OPV2V+**   | CoPerception+: Extending OPV2V for Multi‐Vehicle Fusion                               | CVPR (2023)        | [[paper](https://arxiv.org/abs/2303.13560)] [[code](https://github.com/MediaBrain-SJTU/CoCa3D)] [[project](https://siheng-chen.github.io/dataset/CoPerception+)]                        |
| **V2V4Real** | V2V4Real: A Real‐World V2V Cooperative Perception Dataset                             | CVPR (2023)        | [[paper](https://arxiv.org/abs/2303.07601)] [[code](https://github.com/ucla-mobility/V2V4Real)] [[project](https://mobility-lab.seas.ucla.edu/v2v4real)]                                  |
| **LUCOOP**   | LUCOOP: Leibniz University Cooperative Perception & Urban Navigation Dataset          | IV (2023)          | [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10186693)] [[~~code~~]()] [[project](https://data.uni-hannover.de/dataset/lucoop-leibniz-university-cooperative-perception-and-urban-navigation-dataset )]   |
| **MARS**     | MARS: A Multi‐Agent Relational Simulation Dataset for Cooperative Perception          | CVPR (2024)        | [[paper](https://arxiv.org/abs/2406.09383)] [[code](https://github.com/ai4ce/MARS)] [[project](https://ai4ce.github.io/MARS)]                                                |
| **OPV2V-H**  | HEAL: An Extensible Cooperative Perception Benchmark based on OPV2V                   | ICLR (2024)        | [[paper&review](https://openreview.net/forum?id=KkrDUGIASk)] [[code](https://github.com/yifanlu0227/HEAL)] [[project](https://huggingface.co/datasets/yifanlu/OPV2V-H)]                                          |
| **V2V-QA**   | V2V-QA: A Dataset for Query‐Driven Cooperative Perception                             | arXiv (2025)       | [[paper](https://arxiv.org/abs/2502.09980)] [[code](https://github.com/eddyhkchiu/V2VLLM)] [[project](https://eddyhkchiu.github.io/v2vllm.github.io)]                               |

---


## V2I Datasets <a id="v2i-datasets-anchor"></a>

- **V2I Datasets**: These datasets involve communication between vehicles and infrastructure, supporting cooperative tasks like object detection, tracking, and decision-making in connected environments.

| Acronym            | Paper Title                                                                   | Venue (Year)     | Link                                                                          |
|--------------------|-------------------------------------------------------------------------------|------------------|-------------------------------------------------------------------------------|
| **DAIR-V2X-C**     | DAIR-V2X-C: A Camera–LiDAR Cooperative Perception Dataset                     | CVPR (2022)      |  [[paper](https://arxiv.org/abs/2204.05575)] [[code](https://github.com/AIR-THU/DAIR-V2X)] [[project](https://air.tsinghua.edu.cn/DAIR-V2X/index.html)]                               |
| **V2X-Seq**        | DAIR-V2X-Seq: A Sequence Dataset for V2X Cooperative Perception              | CVPR (2023)      |  [[paper](https://arxiv.org/abs/2305.05938)] [[code](https://github.com/AIR-THU/DAIR-V2X-Seq)] [[project](https://thudair.baai.ac.cn/index)]                                     |
| **HoloVIC**        | HoloVIC: A High-Fidelity Camera–LiDAR Infrastructure Dataset                  | CVPR (2024)      | [[paper](https://arxiv.org/abs/2403.02640)] [~~code~~] [[project](https://holovic.net)]                                                          |
| **OTVIC**          | OTVIC: Online Traffic-side Vehicle Infrastructure Cooperative Dataset         | IROS (2024)      | [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10802656)] [[~~code~~]()] [[project](https://sites.google.com/view/otvic)]                                            |
| **DAIR-V2XReid**   | DAIR-V2XReid: A Cooperative Re-ID Benchmark for V2X                            | T-ITS (2024)     | [[paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10462015)] [[~~code~~]()] [[project](https://github.com/Niuyaqing/DAIR-V2XReid)]                                      |
| **TUMTraf V2X**    | TUMTraf V2X: A Traffic-Side Multi-Modal Cooperative Dataset                    | CVPR (2024)      | [[paper](https://arxiv.org/abs/2403.01316)] [[code](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)] [[project](https://tum-traffic-dataset.github.io/tumtraf-v2x)]                            |
| **V2X-Radar**      | V2X-Radar: A Camera–LiDAR–Radar Cooperative Perception Dataset                 | arXiv (2024)     | [[paper](https://arxiv.org/abs/2411.10962)] [[code](https://github.com/yanglei18/V2X-Radar)] [[project](http://openmpd.com/column/V2X-Radar)]                                            |

---



## V2X Datasets <a id="v2x-datasets-anchor"></a>

- **V2X Datasets**: Covering vehicle-to-everything communication, these datasets integrate multiple agents such as vehicles, infrastructure, and other environmental elements like drones or pedestrians, enabling research in complex collaborative 



| Acronym        | Paper Title                                                                     | Venue (Year)    | Link                                                                         |
|----------------|---------------------------------------------------------------------------------|-----------------|------------------------------------------------------------------------------|
| **V2X-Sim 2.0** | V2X-Sim 2.0: A Synthetic Cooperative Perception Benchmark                        | RA-L (2022)     | [[paper](https://arxiv.org/abs/2202.08449)] [[code](https://github.com/ai4ce/V2X-Sim)] [[project](https://ai4ce.github.io/V2X-Sim)]                                            |
| **V2XSet**     | V2XSet: Multi‐Agent Cooperative Perception in Simulation                         | ECCV (2022)     | [[paper](https://arxiv.org/abs/2203.10638)] [[code](https://github.com/DerrickXuNu/v2x-vit)] [[project](https://paperswithcode.com/dataset/v2xset)]                                   |
| **DOLPHINS**   | DOLPHINS: A Multi-View 2D/3D Cooperative Perception Dataset                      | ACCV (2022)     | [[paper](https://arxiv.org/abs/2207.07609)] [[code](https://github.com/explosion5/Dolphins)] [[project](https://dolphins-dataset.net)]                                                |
| **DeepAccident** | DeepAccident: A Synthetic Traffic Accident Dataset for Cooperative Perception   | AAAI (2024)     | [[paper](https://arxiv.org/abs/2304.01168)] [[code](https://github.com/tianqi-wang1996/DeepAccident)] [[project](https://deepaccident.github.io)]                                              |
| **V2X-Real**   | V2X-Real: A Real-World Multi-Agent Cooperative Perception Dataset                | ECCV (2024)     | [[paper](https://arxiv.org/abs/2403.16034)] [~~code~~] [[project](https://mobility-lab.seas.ucla.edu/v2x-real)]                                  |
| **Multi-V2X**  | Multi-V2X: A Large-Scale Synthetic Cooperative Dataset                           | arXiv (2024)    | [[paper](https://arxiv.org/abs/2409.04980)] [[code](https://github.com/RadetzkyLi/Multi-V2X)] [~~project~~]                                       |
| **Adver-City** | Adver-City: An Adversarial Traffic Simulation Dataset for Cooperative Perception | arXiv (2024)    | [[paper](https://arxiv.org/abs/2410.06380)] [[code](https://github.com/QUARRG/Adver-City)] [[project](https://labs.cs.queensu.ca/quarrg/datasets/adver-city)]                        |
| **V2X-Traj**   | V2X-Traj: A Trajectory Forecasting Benchmark under Cooperative Perception         | NeurIPS (2024)  | [[paper](https://arxiv.org/abs/2311.00371)] [[code](https://github.com/AIR-THU/V2X-Graph)] [[project](https://thudair.baai.ac.cn/index)]                                          |
| **WHALES**     | WHALES: A Whale-Scale Synthetic Dataset for V2X                                    | arXiv (2024)    | [[paper](https://arxiv.org/abs/2411.13340)] [[code](https://github.com/chensiweiTHU/WHALES)] [[project](https://pan.baidu.com/s/1dintX-d1T-m2uACqDlAM9A)]                                       |
| **V2X-R**      | V2X-R: A Radar-Enhanced Cooperative Perception Dataset                           | arXiv (2024)    | [[paper](https://arxiv.org/abs/2411.08402)] [[code](https://github.com/ylwhxht/V2X-R)] [~~project~~]                                             |
| **V2XPnP**     | V2XPnP: Planning-and-Perception for Cooperative Driving                           | arXiv (2024)    | [[paper](https://arxiv.org/abs/2412.01812)] [[code](https://github.com/Zewei-Zhou/V2XPnP)] [[project](https://mobility-lab.seas.ucla.edu/v2xpnp)]                                 |
| **SCOPE**      | SCOPE: Synthetic Cooperative Perception Environment                               | arXiv (2024)    | [[paper](https://arxiv.org/pdf/2408.03065)] [~~code~~] [[project](https://ekut-es.github.io/scope/)]                                             |
| **Mixed Signals** | Mixed Signals: A Real‐World LiDAR‐Only V2X Dataset                             | arXiv (2025)    | [[paper](https://arxiv.org/abs/2502.14156)] [[code](https://github.com/chinitaberrio/Mixed-Signals)] [[project](https://mixedsignalsdataset.cs.cornell.edu)]                                  |

---



## I2I Datasets <a id="i2i-datasets-anchor"></a>

- **I2I Datasets**: Focused on infrastructure-to-infrastructure collaboration, these datasets support research in scenarios with overlapping sensor coverage or distributed sensor fusion across intersections.

| Acronym         | Paper Title                                                                                          | Venue (Year)   | Link                                                                                 |
|-----------------|------------------------------------------------------------------------------------------------------|----------------|--------------------------------------------------------------------------------------|
| **Rcooper**     | Rcooper: DAIR-Rcooper Infrastructure–Vehicle Cooperative Dataset                                       | CVPR (2024)    |  [[paper](https://arxiv.org/abs/2403.10145)] [[code](https://github.com/AIR-THU/DAIR-RCooper)] [[project](https://www.t3caic.com/qingzhen)]                                        |
| **InScope**     | InScope: A LiDAR-Only Infrastructure Cooperative Perception Benchmark                                  | arXiv (2024)   | [[paper](https://arxiv.org/abs/2407.21581)] [[code](https://github.com/xf-zh/InScope)] [~~project~~]                                                    |

---

