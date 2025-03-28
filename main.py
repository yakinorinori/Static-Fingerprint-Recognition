import cv2
import mediapipe as mp
import numpy as np
import math
import pickle
import csv
from collections import defaultdict

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pi = math.pi

# landmarkの繋がり表示用
landmark_line_ids = [
    (0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 0),  # 掌
    (1, 2), (2, 3), (3, 4),         # 親指
    (5, 6), (6, 7), (7, 8),         # 人差し指
    (9, 10), (10, 11), (11, 12),    # 中指
    (13, 14), (14, 15), (15, 16),   # 薬指
    (17, 18), (18, 19), (19, 20),   # 小指
]

count_hand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
hand_moji = ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'さ', 'し', 'す', 'せ', 'そ', 'た', 'ち', 'つ', 'て', 'と', 'な', 'に', 'ぬ', 'ね', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ま', 'み', 'む', 'め', 'や', 'ゆ', 'よ', 'ら', 'る', 'れ', 'ろ', 'わ']
input_characters = []
num = 0

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,                    # 最大検出数
    min_detection_confidence=0.7,       # 検出信頼度
    min_tracking_confidence=0.7         # 追跡信頼度
)
pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
)

directory = 'ans_Fingerspelling/'
with open('hand.pickle', mode='rb') as f:
    svc = pickle.load(f)
src = cv2.imread("入力.png")
src_dict = {key: cv2.imread(f"{directory}{key}.png") for key in hand_moji}

def get_point_coordinates(landmark, img_w, img_h):
    return int(landmark.x * img_w), int(landmark.y * img_h)

def get_distance_squared(point1, point2):
    return ((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2)
def split_on_three_consecutive(input_characters):
    current_char = None
    count = 0
    temp_list = []
    final_list = []

    for char in input_characters:
        if char == current_char:
            count += 1
        else:
            count = 1
            current_char = char
        temp_list.append(char)

        if count == 3:
            final_list.append(temp_list)
            temp_list = []
            count = 0

    if temp_list:
        final_list.append(temp_list)

    return final_list

def dp_matching(frame_data):
    # 初期化
    dp = [[0 for _ in range(len(frame_data[0]))] for _ in range(len(frame_data))]
    dp[0] = frame_data[0]

    # DPテーブルの更新
    for i in range(1, len(frame_data)):
        for j in range(len(frame_data[i])):
            dp[i][j] = max(dp[i-1]) + frame_data[i][j]

    # 後ろから追跡して最適なパスを求める
    path = []
    max_index = dp[-1].index(max(dp[-1]))
    path.append(max_index)

    for i in range(len(dp)-2, -1, -1):
        path.append(dp[i].index(max(dp[i])))

    path = path[::-1]  # パスを逆順にする

    return path

cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        # カメラから画像を取得
        success, img = cap.read()
        if not success:
            continue

        img = cv2.flip(img, 1)              # 左右を反転
        img_h, img_w, _ = img.shape         # サイズ取得

        img.flags.writeable = False
        results_pose = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.flags.writeable = True

        # 検出処理の実行
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            count = 0
            # 検出した手の数分繰り返し
            for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # landmarkのつながりをlineで表示

                # - それぞれの指の定義
                # - それぞれの指の定義
                WRIST = hand_landmarks.landmark[0]  # - 手のひら
                WRIST_x = int(WRIST.x * img_w)
                WRIST_y = int(WRIST.y * img_h)

                # 基準点
                THUMB_MCP = hand_landmarks.landmark[1]
                THUMB_MCP_x = int(THUMB_MCP.x * img_w)
                THUMB_MCP_y = int(THUMB_MCP.y * img_h)

                INDEX_FINGER_MCP = hand_landmarks.landmark[5]
                INDEX_FINGER_MCP_x = int(INDEX_FINGER_MCP.x * img_w)
                INDEX_FINGER_MCP_y = int(INDEX_FINGER_MCP.y * img_h)

                MIDDLE_FINGER_MCP = hand_landmarks.landmark[9]
                MIDDLE_FINGER_MCP_x = int(MIDDLE_FINGER_MCP.x * img_w)
                MIDDLE_FINGER_MCP_y = int(MIDDLE_FINGER_MCP.y * img_h)

                RING_FINGER_MCP = hand_landmarks.landmark[13]
                RING_FINGER_MCP_x = int(RING_FINGER_MCP.x * img_w)
                RING_FINGER_MCP_y = int(RING_FINGER_MCP.y * img_h)

                PINKY_MCP = hand_landmarks.landmark[17]
                PINKY_MCP_x = int(PINKY_MCP.x * img_w)
                PINKY_MCP_y = int(PINKY_MCP.y * img_h)
                # 親指基準点
                ReferencePoint_1 = ((WRIST_x - THUMB_MCP_x) * (WRIST_x - THUMB_MCP_x)) + (
                        (WRIST_y - THUMB_MCP_y) * (WRIST_y - THUMB_MCP_y))
                # 人差し指基準点
                ReferencePoint_2 = ((WRIST_x - INDEX_FINGER_MCP_x) * (WRIST_x - INDEX_FINGER_MCP_x)) + (
                        (WRIST_y - INDEX_FINGER_MCP_y) * (WRIST_y - INDEX_FINGER_MCP_y))
                # 中指基準点
                ReferencePoint_3 = ((WRIST_x - MIDDLE_FINGER_MCP_x) * (WRIST_x - MIDDLE_FINGER_MCP_x)) + (
                        (WRIST_y - MIDDLE_FINGER_MCP_y) * (WRIST_y - MIDDLE_FINGER_MCP_y))
                # 薬指基準点
                ReferencePoint_4 = ((WRIST_x - RING_FINGER_MCP_x) * (WRIST_x - RING_FINGER_MCP_x)) + (
                        (WRIST_y - RING_FINGER_MCP_y) * (WRIST_y - RING_FINGER_MCP_y))
                # 小指基準点
                ReferencePoint_5 = ((WRIST_x - INDEX_FINGER_MCP_x) * (WRIST_x - INDEX_FINGER_MCP_x)) + (
                        (WRIST_y - PINKY_MCP_y) * (WRIST_y - PINKY_MCP_y))
                ReferencePoint_Y = ((INDEX_FINGER_MCP_x - PINKY_MCP_x) * (INDEX_FINGER_MCP_x - PINKY_MCP_x)) + (
                        (INDEX_FINGER_MCP_y - PINKY_MCP_y) * (INDEX_FINGER_MCP_y - PINKY_MCP_y))

                THUMB_IP = hand_landmarks.landmark[3]  # - 親指
                THUMB_IP_x = int(THUMB_IP.x * img_w)
                THUMB_IP_y = int(THUMB_IP.y * img_h)
                THUMB_TIP = hand_landmarks.landmark[4]
                THUMB_TIP_x = int(THUMB_TIP.x * img_w)
                THUMB_TIP_y = int(THUMB_TIP.y * img_h)
                THUMB_TIP_z = int(THUMB_TIP.z)

                INDEX_FINGER_PIP = hand_landmarks.landmark[7]  # - 人差し指
                INDEX_FINGER_PIP_x = int(INDEX_FINGER_PIP.x * img_w)
                INDEX_FINGER_PIP_y = int(INDEX_FINGER_PIP.y * img_h)
                INDEX_FINGER_TIP = hand_landmarks.landmark[8]
                INDEX_FINGER_TIP_x = int(INDEX_FINGER_TIP.x * img_w)
                INDEX_FINGER_TIP_y = int(INDEX_FINGER_TIP.y * img_h)
                INDEX_FINGER_TIP_z = int(INDEX_FINGER_TIP.z)

                MIDDLE_FINGER_PIP = hand_landmarks.landmark[11]  # - 中指
                MIDDLE_FINGER_PIP_x = int(MIDDLE_FINGER_PIP.x * img_w)
                MIDDLE_FINGER_PIP_y = int(MIDDLE_FINGER_PIP.y * img_h)
                MIDDLE_FINGER_TIP = hand_landmarks.landmark[12]
                MIDDLE_FINGER_TIP_x = int(MIDDLE_FINGER_TIP.x * img_w)
                MIDDLE_FINGER_TIP_y = int(MIDDLE_FINGER_TIP.y * img_h)
                MIDDLE_FINGER_TIP_z = int(MIDDLE_FINGER_TIP.z)

                RING_FINGER_PIP = hand_landmarks.landmark[15]  # - 薬指
                RING_FINGER_PIP_x = int(RING_FINGER_PIP.x * img_w)
                RING_FINGER_PIP_y = int(RING_FINGER_PIP.y * img_h)
                RING_FINGER_TIP = hand_landmarks.landmark[16]
                RING_FINGER_TIP_x = int(RING_FINGER_TIP.x * img_w)
                RING_FINGER_TIP_y = int(RING_FINGER_TIP.y * img_h)
                RING_FINGER_TIP_z = int(RING_FINGER_TIP.z)

                PINKY_PIP = hand_landmarks.landmark[19]  # - 小指
                PINKY_PIP_x = int(PINKY_PIP.x * img_w)
                PINKY_PIP_y = int(PINKY_PIP.y * img_h)
                PINKY_TIP = hand_landmarks.landmark[20]
                PINKY_TIP_x = int(PINKY_TIP.x * img_w)
                PINKY_TIP_y = int(PINKY_TIP.y * img_h)
                PINKY_TIP_z = int(PINKY_TIP.z)

                JUDGE_THUMB_1 = ((WRIST_x - THUMB_IP_x) * (WRIST_x - THUMB_IP_x)) + (
                        (WRIST_y - THUMB_IP_y) * (WRIST_y - THUMB_IP_y))  # - 親指
                JUDGE_THUMB_2 = ((WRIST_x - THUMB_TIP_x) * (WRIST_x - THUMB_TIP_x)) + (
                        (WRIST_y - THUMB_TIP_y) * (WRIST_y - THUMB_TIP_y))

                JUDGE_INDEX_FINGER_1 = ((WRIST_x - INDEX_FINGER_PIP_x) * (WRIST_x - INDEX_FINGER_PIP_x)) + (
                        (WRIST_y - INDEX_FINGER_PIP_y) * (WRIST_y - INDEX_FINGER_PIP_y))  # - 人差し指
                JUDGE_INDEX_FINGER_2 = ((WRIST_x - INDEX_FINGER_TIP_x) * (WRIST_x - INDEX_FINGER_TIP_x)) + (
                        (WRIST_y - INDEX_FINGER_TIP_y) * (WRIST_y - INDEX_FINGER_TIP_y))

                JUDGE_MIDDLE_FINGER_1 = ((WRIST_x - MIDDLE_FINGER_PIP_x) * (WRIST_x - MIDDLE_FINGER_PIP_x)) + (
                        (WRIST_y - MIDDLE_FINGER_PIP_y) * (WRIST_y - MIDDLE_FINGER_PIP_y))  # - 中指
                JUDGE_MIDDLE_FINGER_2 = ((WRIST_x - MIDDLE_FINGER_TIP_x) * (WRIST_x - MIDDLE_FINGER_TIP_x)) + (
                        (WRIST_y - MIDDLE_FINGER_TIP_y) * (WRIST_y - MIDDLE_FINGER_TIP_y))

                JUDGE_RING_FINGER_1 = ((WRIST_x - RING_FINGER_PIP_x) * (WRIST_x - RING_FINGER_PIP_x)) + (
                        (WRIST_y - RING_FINGER_PIP_y) * (WRIST_y - RING_FINGER_PIP_y))  # - 薬指
                JUDGE_RING_FINGER_2 = ((WRIST_x - RING_FINGER_TIP_x) * (WRIST_x - RING_FINGER_TIP_x)) + (
                        (WRIST_y - RING_FINGER_TIP_y) * (WRIST_y - RING_FINGER_TIP_y))

                JUDGE_PINKY_1 = ((WRIST_x - PINKY_PIP_x) * (WRIST_x - PINKY_PIP_x)) + (
                        (WRIST_y - PINKY_PIP_y) * (WRIST_y - PINKY_PIP_y))  # - 小指
                JUDGE_PINKY_2 = ((WRIST_x - PINKY_TIP_x) * (WRIST_x - PINKY_TIP_x)) + (
                        (WRIST_y - PINKY_TIP_y) * (WRIST_y - PINKY_TIP_y))

                JUDGE_THUMB_INDEX = ((THUMB_TIP_x - INDEX_FINGER_TIP_x) * (THUMB_TIP_x - INDEX_FINGER_TIP_x)) + (
                        (THUMB_TIP_y - INDEX_FINGER_TIP_y) * (THUMB_TIP_y - INDEX_FINGER_TIP_y))  # - 親指と人差し指の長さ
                JUDGE_THUMB_MIDDLE = ((THUMB_TIP_x - MIDDLE_FINGER_TIP_x) * (THUMB_TIP_x - MIDDLE_FINGER_TIP_x)) + (
                        (THUMB_TIP_y - MIDDLE_FINGER_TIP_y) * (THUMB_TIP_y - MIDDLE_FINGER_TIP_y))  # - 親指と中指の長さ
                JUDGE_THUMB_RING = ((THUMB_TIP_x - RING_FINGER_TIP_x) * (THUMB_TIP_x - RING_FINGER_TIP_x)) + (
                        (THUMB_TIP_y - RING_FINGER_TIP_y) * (THUMB_TIP_y - RING_FINGER_TIP_y))  # - 親指と薬指の長さ
                JUDGE_THUMB_PINKY = ((THUMB_TIP_x - PINKY_TIP_x) * (THUMB_TIP_x - PINKY_TIP_x)) + (
                        (THUMB_TIP_y - PINKY_TIP_y) * (THUMB_TIP_y - PINKY_TIP_y))  # - 親指と小指の長さ
                # 隣の指同士の距離
                NEXT_INDEX_MIDDLE = ((INDEX_FINGER_TIP_x - MIDDLE_FINGER_TIP_x) * (
                        INDEX_FINGER_TIP_x - MIDDLE_FINGER_TIP_x)) + (
                                            (INDEX_FINGER_TIP_y - MIDDLE_FINGER_TIP_y) * (
                                            INDEX_FINGER_TIP_y - MIDDLE_FINGER_TIP_y))
                NEXT_MIDDLE_RING = ((MIDDLE_FINGER_TIP_x - RING_FINGER_TIP_x) * (
                        MIDDLE_FINGER_TIP_x - RING_FINGER_TIP_x)) + (
                                           (MIDDLE_FINGER_TIP_y - RING_FINGER_TIP_y) * (
                                           MIDDLE_FINGER_TIP_y - RING_FINGER_TIP_y))
                NEXT_RING_PINKY = ((RING_FINGER_TIP_x - PINKY_TIP_x) * (RING_FINGER_TIP_x - PINKY_TIP_x)) + (
                        (RING_FINGER_TIP_y - PINKY_TIP_y) * (RING_FINGER_TIP_y - PINKY_TIP_y))

                for line_id in landmark_line_ids:
                    # 1点目の座標取得
                    lm = hand_landmarks.landmark[line_id[0]]
                    lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))

                    if lm == hand_landmarks.landmark[3] or lm == hand_landmarks.landmark[4] or lm == \
                            hand_landmarks.landmark[7] or lm == hand_landmarks.landmark[8] or lm == \
                            hand_landmarks.landmark[11] or lm == hand_landmarks.landmark[12] or lm == \
                            hand_landmarks.landmark[15] or lm == hand_landmarks.landmark[16] or lm == \
                            hand_landmarks.landmark[19] or lm == hand_landmarks.landmark[20]:
                        cv2.circle(img, lm_pos1, 10, (0, 0, 255), thickness=-1)
                    else:
                        cv2.circle(img, lm_pos1, 7, (255, 255, 255), thickness=-1)

                    # 2点目の座標取得
                    lm = hand_landmarks.landmark[line_id[1]]
                    lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
                    if lm == hand_landmarks.landmark[3] or lm == hand_landmarks.landmark[4] or lm == \
                            hand_landmarks.landmark[7] or lm == hand_landmarks.landmark[8] or lm == \
                            hand_landmarks.landmark[11] or lm == hand_landmarks.landmark[12] or lm == \
                            hand_landmarks.landmark[15] or lm == hand_landmarks.landmark[16] or lm == \
                            hand_landmarks.landmark[19] or lm == hand_landmarks.landmark[20]:
                        cv2.circle(img, lm_pos2, 10, (0, 0, 255), thickness=-1)
                    else:
                        cv2.circle(img, lm_pos2, 7, (255, 255, 255), thickness=-1)

                    cv2.line(img, lm_pos1, lm_pos2, (255, 255, 255), 3)

                    hand_texts1 = []  # - 座標表示
                    for c_id, hand_class in enumerate(results.multi_handedness[h_id].classification):
                        # - テキスト表示に必要な座標など準備
                        lm = hand_landmarks.landmark[0]
                        lm_x = int(lm.x * img_w) - 120
                        lm_y = int(lm.y * img_h) + 30
                        lm_c = (64, 0, 0)
                        X1 = (int(lm.x * img_w), int(lm.y * img_h))
                        lm_xy = hand_landmarks.landmark[9]  # 基準点(角度用)
                        lm_xy_1 = hand_landmarks.landmark[4]
                        lm_xy_2 = hand_landmarks.landmark[8]
                        lm_xy_3 = hand_landmarks.landmark[12]
                        lm_xy_4 = hand_landmarks.landmark[16]
                        lm_xy_5 = hand_landmarks.landmark[20]
                        lm_xy_6 = hand_landmarks.landmark[3]
                        lm_xy_7 = hand_landmarks.landmark[10]
                        lm_xy_8 = hand_landmarks.landmark[18]
                        lm_xy_9 = hand_landmarks.landmark[6]
                        lm_xy_10 = hand_landmarks.landmark[14]

                        xy_1 = np.arctan2(lm_xy_1.x * img_w - lm_xy.x * img_w, lm_xy_1.y * img_h - lm_xy.y * img_h)
                        xy_2 = np.arctan2(lm_xy_2.x * img_w - lm_xy.x * img_w, lm_xy_2.y * img_h - lm_xy.y * img_h)
                        xy_3 = np.arctan2(lm_xy_3.x * img_w - lm_xy.x * img_w, lm_xy_3.y * img_h - lm_xy.y * img_h)
                        xy_4 = np.arctan2(lm_xy_4.x * img_w - lm_xy.x * img_w, lm_xy_4.y * img_h - lm_xy.y * img_h)
                        xy_5 = np.arctan2(lm_xy_5.x * img_w - lm_xy.x * img_w, lm_xy_5.y * img_h - lm_xy.y * img_h)
                        xy_6 = np.arctan2(lm_xy_1.x * img_w - lm_xy_6.x * img_w, lm_xy_1.y * img_h - lm_xy_6.y * img_h)
                        xy_7 = np.arctan2(lm_xy_3.x * img_w - lm_xy_7.x * img_w, lm_xy_3.y * img_h - lm_xy_7.y * img_h)
                        xy_8 = np.arctan2(lm_xy_5.x * img_w - lm_xy_8.x * img_w, lm_xy_5.y * img_h - lm_xy_8.y * img_h)
                        xy_9 = np.arctan2(lm_xy_2.x * img_w - lm_xy_9.x * img_w, lm_xy_2.y * img_h - lm_xy_9.y * img_h)
                        xy_10 = np.arctan2(lm_xy_4.x * img_w - lm_xy_10.x * img_w, lm_xy_4.y * img_h - lm_xy_10.y * img_h)
                        a = np.arctan2(lm_xy.x * img_w - lm.x * img_w, lm_xy.y * img_h - lm.y * img_h)
                        b = np.arctan2(lm_xy_1.x * img_w - lm.x * img_w, lm_xy_1.z * img_w - lm.z * img_w)
                        c = np.arctan2(lm_xy_5.x * img_w - lm.x * img_w, lm_xy_5.z * img_w - lm.z * img_w)
                        font = cv2.FONT_HERSHEY_SIMPLEX

                num += 1

                hand_data = [
                    [JUDGE_THUMB_1 / ReferencePoint_1,
                     (JUDGE_THUMB_1 / ReferencePoint_1) - (JUDGE_THUMB_2 / ReferencePoint_1),
                     JUDGE_INDEX_FINGER_1 / ReferencePoint_2,
                     (JUDGE_INDEX_FINGER_1 / ReferencePoint_2) - (JUDGE_INDEX_FINGER_2 / ReferencePoint_2),
                     JUDGE_MIDDLE_FINGER_1 / ReferencePoint_3,
                     (JUDGE_MIDDLE_FINGER_1 / ReferencePoint_3) - (JUDGE_MIDDLE_FINGER_2 / ReferencePoint_3),
                     JUDGE_RING_FINGER_1 / ReferencePoint_4,
                     (JUDGE_RING_FINGER_1 / ReferencePoint_4) - (JUDGE_RING_FINGER_2 / ReferencePoint_4),
                     JUDGE_PINKY_1 / ReferencePoint_5 * 2,
                     ((JUDGE_PINKY_1 / ReferencePoint_5) - (JUDGE_PINKY_2 / ReferencePoint_5)) * 2,
                     JUDGE_THUMB_INDEX / ReferencePoint_Y,
                     JUDGE_THUMB_MIDDLE / ReferencePoint_Y, JUDGE_THUMB_RING / ReferencePoint_Y,
                     JUDGE_THUMB_PINKY / ReferencePoint_Y, NEXT_INDEX_MIDDLE / ReferencePoint_Y,
                     NEXT_MIDDLE_RING / ReferencePoint_Y, NEXT_RING_PINKY / ReferencePoint_Y,
                     a, b, c, xy_1, xy_2, xy_3, xy_4, xy_5, xy_6, xy_7, xy_8, xy_9, xy_10]
                ]

                ans = svc.predict(hand_data)

                # 手話の文字を表現する座標データ
                input_data = hand_data

                # リストをnumpy配列に変換
                input_data = np.array(input_data)

                # 所属確率の計算
                probabilities = svc.predict_proba(input_data.reshape(1, -1))

                # 所属確率とクラス名を対応させる
                prob_with_classes = list(zip(svc.classes_, probabilities[0]))

                # 所属確率の降順に並び替え
                prob_with_classes.sort(key=lambda x: x[1], reverse=True)

                # 保持するランキング文字のリスト
                input_chars_rankings = []  # 追加: 初期化を行います

                # 保持するランキング文字のリスト
                input_ranking_chars = [item[0] for item in prob_with_classes[:5]]
                input_chars_rankings.append(input_ranking_chars)

                f = open('jikken.csv', mode="a", newline="")
                writer = csv.writer(f)

                # 最も可能性が高い5つのクラスを表示
                writer.writerow([f'{num}フレーム目'])
                for i in range(5):
                    output_string = f'{i + 1}位 文字:'
                    output_string1 = f'「{prob_with_classes[i][0]}」'
                    output_string2 = f'{prob_with_classes[i][1] * 100} %'
                    print(output_string, output_string2)
                    writer.writerow([output_string, output_string1, output_string2])
                f.close()

                grouped_characters = []
                current_group = []
                input_characters.append(ans)

                for char_list in input_characters:
                    char = char_list[0]
                    if not current_group or char == current_group[0]:
                        current_group.append(char)
                    else:
                        if len(current_group) >= 15:
                            grouped_characters.append(current_group[0])

                            # 各ランキングの平均登場回数を計算
                            for rank in range(5):
                                occurrences = [chars[rank] for chars in input_chars_rankings[-15:]].count(
                                    grouped_characters[-1])
                                print(
                                    f"Average occurrence of ‘{grouped_characters[-1]}’ at rank {rank + 1} over the last 15 frames: {occurrences} times")

                            input_chars_rankings = []  # ランキングリストをリセット

                        current_group = [char]

                if current_group and len(current_group) >= 10:
                    grouped_characters.append(current_group[0])

                print(grouped_characters)

                text = ''.join(grouped_characters)

                test_result = sorted(count_hand, reverse=True)

                if grouped_characters:
                    for idx, character in enumerate(grouped_characters):
                        char_str = str(character)
                        if char_str in src_dict:
                            window_name = f"grouped_answer_{idx}"
                            cv2.imshow(window_name, src_dict[char_str])
                            cv2.moveWindow(window_name, 0 + 250 * idx, 0)

                ans_str = str(ans[0])
                if ans_str in src_dict:
                    cv2.imshow("answer", src_dict[ans_str])
                    cv2.moveWindow("answer", 1000, 0)

                    # 初期化
                    count_hand = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    num = 0


        #  画像の表示
        cv2.imshow("MediaPipe Hands", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q') or key == 0x1b:
            break

cap.release()
