# Nvidia-AI-Specialist-Cirtification
# 제목(Title)
AI 객체 인식을 활용한 표지판 기반 주행 제어 

Signboard-based driving control using AI object recognition
___
# 개요(Overview)
운전자와 보행자들이 안전하고 효율적으로 도로를 이용하려면 도료 표지판의 식별과 해석이 중요하다. 그러나 길가에는 수많은 표지판들이 있기 때문에 인간의 판단에 의존하는 경우 실수나 위험이 발생할 수 있다. 이를 해결하기 위해, AI 객체 인식을 활용하여 도로 표지판을 탐지하고 분류하는 시스템을 개발하고자 한다.

최근 객체 인식 기술과 AI 기술이 발전하면서 자율 주행, 운전자 보조 시스템, 교통 관리 시스템, 스마트 내비게이션, 교통법 위반 감지 시스템과 같이 많은 것들이 가능해졌다. 이번 프로젝트는 그에 필요한 시스템인 도로 표지판의 형태와 종류를 정확히 인식하고 분류하는 시스템을 YOLOv5를 사용하여 실현하는 것에 중점을 둔다.

The identification and interpretation of paint signs are important for drivers and pedestrians to use the road safely and efficiently. However, since there are numerous signs on the road, mistakes or risks can occur if they rely on human judgment. To solve this problem, we intend to develop a system that detects and classifies road signs using AI object recognition.

With the recent development of object recognition technology and AI technology, many things have become possible such as autonomous driving, driver assistance systems, traffic management systems, smart navigation systems, and traffic law violation detection systems. This project focuses on realizing a system that accurately recognizes and classifies the shape and type of road signs, which are necessary for them, using YOLOv5.
___
# 영상 취득 방법(Video Acquisition Method)

영상은 고속도로를 주행하며 표지판이 나오면 촬영을 했다. 영상은 총 두 개이며 video 1은 AI 학습에 사용했고, verify 1은 검증하는데 사용했다.

The video was filmed when the sign appeared while driving on the highway. There are a total of two videos, and video 1 was used for AI learning and verification 1 was used for verification.

학습 비디오()
https://www.youtube.com/watch?v=28CcGXX3-3A

검증 비디오()
https://www.youtube.com/watch?v=Tk15Pvsrla8
