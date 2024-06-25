제품 유무 분류기 작동법

아나콘다로 가상환경 생성 추천(파이썬 venv로도 가상환경 생성해서 진행해도 상관없음)
$ conda create -n yolonas python=3.9.17 -y

가상환경 활성화
$ conda activate yolonas
(yolonas) <- 가상환경 실행된 후 가상환경이 작동됐는지 확인

필수 라이브러리 설치
$ pip install torch torchvision super-gradients opencv-python tqdm pycocotools openpyxl

분류할 이미지 폴더를 detect_init.py가 있는 곳으로 옮긴후 폴더명 및 파일명에 꼭 영어가 아닌 언어가 없도록 해야함
이미지 폴더를 옮긴 후 폴더의 구조는 아래와 같아야함
detect
ㄴ {image_root_folder} # 수집한 키워드명
  ㄴ {image_child_folder} # 수집한 UID
    ㄴ {image_file}.jpg # 영상에서 프레임분할된 이미지
ㄴ detect.py
ㄴ obj_ckpt_best.pth
ㄴ 제품클래스.xlsx

따라서, 이미지 폴더의 상위 폴더명은 수집한 키워드명을 추천하며(eg. lipstick), 하위 폴더명은 UID로 고정되어 있기에 따로 수정하지 않았다면 영어가 아닌 언어가 들어갔을 리가 없기에 그대로 두면 됨.

코드 실행
    $ python detect.py -p {image_root_folder} -c {confidence}

    {image_root_folder}에는 이미지 상위 폴더명을 입력하거나 비워두면 됨
    비워둘 시 현재 경로를 자동 탐색하여 작업 진행함

    {confidence}는 디폴트값은 0.7이나 이를 수정하고 싶으면 0~1 범위 내의 수로 수정하면 됨(1에 가까울 수록 모델의 판단이 까다로워지기에 적절한 수가 중요함. 디폴트값이 0.7이 제일 무난함)
    *예시1 만약, lipstick이라는 명의 폴더명을 가진 파일을 confidence 0.8로 분류하고 싶다면 아래와 같이 터미널에 입력하면 됨
        $ python detect.py -p lipstick -c 0.8

    *예시2 만약, lipstick이라는 명의 폴더명을 가진 파일을 confidence 0.7로 분류하고 싶다면 아래와 같이 터미널에 입력하면 됨
        $ python detect.py -p lipstick or $ python detect_init.py -p lipstick -c 0.7
        
    *예시3 만약, 현재 경로를 기준으로 모든 폴더에 작업을 하고 confidence를 0.6으로 분류하고 싶다면 아래와 같이 터미널에 입력하면 됨
        $ python detect.py -c 0.6

완료 후 결과물
    작업이 완료되면 {image_child_folder}에 detected, none, json 폴더가 생성됨
    detected 폴더에는 객체가 탐지된 이미지들이 복사됨
    none 폴더에는 객체가 탐지되지 않은 이미지들이 복사됨
    json 폴더에는 각 파일별 간략한 결과가 저장됨

또한, 전체 결과를 간략하게 표시하기 위해 {image_root_folder}_result.json이라는 파일이 메인 디렉토리에 저장됨