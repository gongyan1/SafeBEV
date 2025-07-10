# SafeBEV3.0
To ensure the safety of autonomous driving perception, beyond onboard perception which is influenced by factors such as vehicle position, sensor location, and object occlusion, roadside perception systems are typically characterized by fixed sensor placements and elevated installation heights, which contribute to a significant reduction in the adverse effects of occlusion on perception reliability and safety. And through cooperative perception methods such as V2V, V2I, and V2X, poses and features obtained from different perception devices can be effectively fused, providing autonomous vehicles with broader-range and more reliable perception results, thereby enhancing the safety of autonomous perception.

![cooperative_method](Fig\05_cooperative_method.png)


## Roadside BEV Perception Methods
### Camera
- BEVHeight: A Robust Framework for Vision-based Roadside 3D Object Detection / [paper](https://arxiv.org/pdf/2303.08498) / [code](https://github.com/ADLab-AutoDrive/BEVHeight) / CVPR2023 / BEVHeight
- BEVHeight++: Toward Robust Visual Centric 3D Object Detection / [paper](https://arxiv.org/pdf/2309.16179) / [code](https://github.com/yanglei18/BEVHeight_Plus) / TPAMI2025 / BEVHeight++
- CoBEV: Elevating Roadside 3D Object Detection with Depth and Height Complementarity / [paper](https://arxiv.org/pdf/2310.02815) / [code](https://github.com/MasterHow/CoBEV) / CTIP2024 / CoBEV
- BEVSpread: Spread Voxel Pooling for Bird's-Eye-View Representation in Vision-based Roadside 3D Object Detection / [paper](https://arxiv.org/pdf/2406.08785) / [code](https://github.com/DaTongjie/BEVSpread) / CVPR2024 / BEVSpread
- Calibration-free BEV Representation for Infrastructure Perception / [paper](https://arxiv.org/pdf/2303.03583) / [code](https://github.com/leofansq/CBR) / IROS2023 / CBR
- RopeBEV: A Multi-Camera Roadside Perception Network in Bird's-Eye-View / [paper](https://arxiv.org/pdf/2409.11706) / arXiv2024 / RopeBEV
### LiDAR
- Roadside Lidar Vehicle Detection and Tracking Using Range And Intensity Background Subtraction / [paper](https://arxiv.org/pdf/2201.04756) / Jorunal of Adavnced Transportation
- Automatic Vehicle Tracking With Roadside LiDAR Data for the Connected-Vehicles System / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8721124) / MIS 2019
- Automatic background filtering and lane identification with roadside LiDAR data / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8317723) / ITSC 2017
- An Automatic Lane Marking Detection Method With Low-Density Roadside LiDAR Data / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9350286) / JSEN 2021
- Center-Aware 3D Object Detection with Attention Mechanism Based on Roadside LiDAR / [paper](https://www.mdpi.com/2071-1050/15/3/2628/pdf?version=1675302134) / Sustainability 2023 / CetrRoad
- Detection and tracking of pedestrians and vehicles using roadside LiDAR sensors / [paper](https://www.sciencedirect.com/science/article/pii/S0968090X19300282/pdfft?md5=bfad0d3a96b656b9bae31e256f441e78&pid=1-s2.0-S0968090X19300282-main.pdf) / TR_C 2019 
### Fusion
- BEVRoad: a Cross-Modal and Temporary-Recurrent 3D Object Detector for Infrastructure Perception / [paper](https://easychair.org/publications/preprint/bj6K/open) / easychair 2024 / BEVRoad
- Accurate and Robust Roadside 3-D Object Detection Based on Height-Aware Scene Reconstruction / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10643791) / JSEN 2024 / HSRDet
- Object Tracking Based on the Fusion of Roadside LiDAR and Camera Data / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9868055) / TIM 2022 

## Collaborative BEV Perception Methods
### V2V
#### Single-Modal
- CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers / [paper](https://arxiv.org/pdf/2207.02202) / [code](https://github.com/DerrickXuNu/CoBEVT) / CoRL 2022 / CoBEVT
- Unlocking Past Information: Temporal Embeddings in Cooperative Bird's Eye View Prediction / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10588608) / [code](https://github.com/cvims/TempCoBE) / IV 2024 / TempCoBEV
- Collaboration Helps Camera Overtake LiDAR in 3D Detection / [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Hu_Collaboration_Helps_Camera_Overtake_LiDAR_in_3D_Detection_CVPR_2023_paper.pdf) / [code](https://github.com/MediaBrain-SJTU/CoCa3D) / CVPR 2023 / CoCa3D
- Asynchrony-Robust Collaborative Perception via Bird's Eye View Flow / [paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/5a829e299ebc1c1615ddb09e98fb6ce8-Paper-Conference.pdf) / [code](https://github.com/MediaBrain-SJTU/CoBEVFlow) / NIPS 2023 / CoBEVFlow
- V2VNet: Vehicle-to-Vehicle Communication for Joint Perception and Predictio / [paper](https://arxiv.org/pdf/2008.07519) / ECCV 2020 / V2VNet
- Learning for Vehicle-to-Vehicle Cooperative Perception under Lossy Communication / [paper](https://arxiv.org/pdf/2212.08273) / [code](https://github.com/jinlong17/V2VLC) / TIV 2023 / LCRN-V2VAM
#### Multi-Modal
- CoBEVFusion: Cooperative Perception with LiDAR-Camera Bird's-Eye View Fusion / [paper](https://arxiv.org/pdf/2310.06008) / DICTA 2024 / CoBEVFusion
- HM-ViT: Hetero-modal Vehicle-to-Vehicle Cooperative perception with vision transformer / [paper](https://arxiv.org/pdf/2304.10628) / [code](https://github.com/XHwind/HM-ViT) / ICCV 2023 / HM-ViT
- MCoT: Multi-Modal Vehicle-to-Vehicle Cooperative Perception with Transformers / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10476067) / ICPADS 2023 / MCoT
- V2VFormer++: Multi-Modal Vehicle-to-Vehicle Cooperative Perception via Global-Local Transformer / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10265751) / TITS 2023 / V2VFormer++
### V2I
#### Single-Modal
- VIMI: Vehicle-Infrastructure Multi-view Intermediate Fusion for Camera-based 3D Object Detection / [paper](https://arxiv.org/pdf/2303.10975) / arXiv 2023 / VIMI
- BEVSync: Asynchronous Data Alignment for Camera-based Vehicle-Infrastructure Cooperative Perception Under Uncertain Delays / [paper](https://ojs.aaai.org/index.php/AAAI/article/view/33611/35766) / AAAI 2025 / BEVSync
- VI-BEV: Vehicle-Infrastructure Collaborative Perception for 3-D Object Detection on Bird's-Eye View / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10896690) / OJITS 2025 / VI-BEV
- CoFormerNet: A Transformer-Based Fusion Approach for Enhanced Vehicle-Infrastructure Cooperative Perception / [paper](https://www.mdpi.com/1424-8220/24/13/4101/pdf?version=1720164678) / sensors 2024 / CoFormerNet
- V2IViewer: Towards Efficient Collaborative Perception via Point Cloud Data Fusion and Vehicle-to-Infrastructure Communications / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10720085) / TNSE 2024 / V2IViewer
- CenterCoop: Center-Based Feature Aggregation for Communication-Efficient Vehicle-Infrastructure Cooperative 3D Object Detection / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10342783) / RAL 2023 / CenterCoop
#### Multi-Modal
- V2I-BEVF: Multi-modal Fusion Based on BEV Representation for Vehicle-Infrastructure Perception / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10421963) / ITSC 2023 / V2I-BEVF
- V2I-Coop: Accurate Object Detection for Connected Automated Vehicles At Accident Black Spots With V2I Cross-Modality Cooperation / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10736566) / TMC 2024 / V2I-Coop
- MSMDFusion: Fusing LiDAR and Camera at Multiple Scales with Multi-Depth Seeds for 3D Object Detection / [paper](https://arxiv.org/pdf/2209.03102) / [code](https://github.com/sxjyjay/msmdfusion) / CVPR 2023 / MSMDFusion
- CO^3: Cooperative Unsupervised 3D Representation Learning for Autonomous Driving / [paper](https://arxiv.org/pdf/2206.04028) / [code](https://github.com/Runjian-Chen/CO3) / ICLR 2023 / CO^3
- Multistage Fusion Approach of Lidar and Camera for Vehicle-Infrastructure Cooperative Object Detection / [paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10021459) / WCMEIM 2022 / VICOD
### I2I&V2X
- V2X-BGN: Camera-based V2X-Collaborative 3D Object Detection with BEV Global Non-Maximum Suppression / [papre](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10588592) / IV 2024 / V2X-BGN
- BEV-V2X: Cooperative Birds-Eye-View Fusion and Grid Occupancy Prediction via V2X-Based Data Sharing / [papre](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10179171) / TIV 2023 / BEV-V2X