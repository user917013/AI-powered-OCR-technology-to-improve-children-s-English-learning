import cv2
import os
import numpy as np
import pytesseract
import openai
import re           #呼叫函數


openai.api_key = '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'   #api_key


def load_image(filename):                                                #翻轉校正
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape[:2]
    if width < height:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    if width > 700 or height > 700:
        scale_factor = min(700 / width, 700 / height)
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

    return img
    

def capture_image():                                                           #相機截圖
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Image', frame)
        if cv2.waitKey(1) != -1: # Press any key to stop capturing
            cv2.imwrite('capture.png', frame)
            break
    cap.release()
    cv2.destroyAllWindows()

    return 'capture.png'


def detect_black_areas(img):                                 #圖像處理
    dst = cv2.add(img, 70)
    thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel =  np.ones((3,3), np.uint8)
    kernel2 = np.ones((7,7), np.uint8)
    kernel3 = np.ones((15,15), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    opening = cv2.erode(opening, kernel2)
    denoised = cv2.dilate(opening, kernel2)

    black_area_percentage_threshold = 5
    contours, hierarchy  = cv2.findContours(denoised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        total_pixels = denoised.shape[0] * denoised.shape[1]
        black_area_percentage = (area / total_pixels) * 100

        if black_area_percentage > black_area_percentage_threshold:
            cv2.drawContours(denoised, [contour], 0, 255, -1)

    denoised = cv2.erode(denoised, kernel3)
    result = denoised - thresh
    result = cv2.erode(result, kernel)
    result = cv2.bitwise_not(result)
    result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel, iterations=3)
    denoised2 = cv2.erode(result, kernel2)
    denoised2 = cv2.dilate(denoised2, kernel)

    return denoised2

    

def extract_text(img):                   #呼叫tesseract辨識文字
     text = pytesseract.image_to_string(img, config='--psm 9')
     return text


def display_images(img, processed_img):      #顯示圖片

    cv2.imshow("Original Image", img)
    cv2.imshow("Processed Image", processed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_text_with_gpt35(text):          #呼叫 GPT-3.5回答
    if text == "":
        return
    else:
        completion = openai.Completion.create(
        engine="text-davinci-003",
        prompt=text,
        max_tokens=100,
        n=1,
        temperature=0.1
    )
    return completion.choices[0].text.strip()

def extract_english(text):                         #處理文字
    english_only = re.findall(r'[a-zA-Z]+', text)  
    result = ''.join(english_only).upper()  
    return result

   
    


if __name__ == "__main__":                 #主程式
    while True:
        choice = input("請選擇輸入圖檔或使用相機截圖（輸入'1'選擇圖檔，輸入'2'選擇相機截圖，輸入'q'退出）: ")
        if choice == '1':
            # Input image file
            filename = input("請輸入檔案名稱: ")
            dir_path = os.getcwd()
            matching_files = [f for f in os.listdir(dir_path) if f.lower().startswith(filename.lower()) and (f.lower().endswith('.png') or f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.bmp'))]
            if len(matching_files) == 0:
                print('找不到符合條件的圖片檔案')
                continue
            if len(matching_files) > 1:
                print('找到多個符合條件的圖片檔案，請輸入完整的檔案名稱')
                continue
            img = load_image(matching_files[0])

            processed_img = detect_black_areas(img)
            text = extract_text(processed_img)
            text = extract_english(text)
            print("偵測到的文字:\n", text)
            display_images(img, processed_img)
            gpt35_result = generate_text_with_gpt35("用中文回答 如果沒有文字不用回答 找可能 " + text + "的意思 ")
            print("GPT-3.5 ：", gpt35_result)
            
        elif choice == '2':
            capture_image()
            img = load_image('capture.png')
            processed_img = detect_black_areas(img)
            text = extract_text(processed_img)
            text = extract_english(text)
            print("偵測到的文字:\n", text)
            display_images(img, processed_img)
            gpt35_result = generate_text_with_gpt35("用中文回答 如果沒有文字不用回答 找可能接近 " + text + "的單字 ")
            print("GPT-3.5 ：", gpt35_result)
            
        elif choice == 'q':
            print("程式已退出。")
            break
        else:
            print("錯誤: 請輸入有效的選項。")
            continue


