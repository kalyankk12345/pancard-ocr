# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import re
import io
import json
import ftfy
import csv
from numpy import unicode

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe"


def ocr_apply(args):
    # load the example image and convert it to grayscale
    image = cv2.imread(args["image"])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # check to see if we should apply thresholding to preprocess the
    # image
    if args["preprocess"] == "thresh":
        gray = cv2.threshold(gray, 0, 255,
                             cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    elif args["preprocess"] == "adaptive":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)

    if args["preprocess"] == "linear":
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    elif args["preprocess"] == "cubic":
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # make a check to see if blurring should be done to remove noise, first is default median blurring

    if args["preprocess"] == "blur":
        gray = cv2.medianBlur(gray, 3)

    elif args["preprocess"] == "bilateral":
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

    elif args["preprocess"] == "gauss":
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('Proccesed',gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # write the grayscale image to disk as a temporary file so we can
    # apply OCR to it
    filename = "{}.png".format(os.getpid())
    cv2.imwrite(filename, gray)

    # load the image as a PIL/Pillow image, apply OCR, and then delete
    # the temporary file
    text = pytesseract.image_to_string(Image.open(filename), lang='eng')
    # add +hin after eng within the same argument to extract hindi specific text - change encoding to utf-8 while writing
    os.remove(filename)
    # print(text)

    # writing extracted data into a text file
    text_output = open('outputbase.txt', 'w', encoding='utf-8')
    text_output.write(text)
    text_output.close()

    file = open('outputbase.txt', 'r', encoding='utf-8')
    text = file.read()
    # print(text)

    # Cleaning all the gibberish text
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)

    # Initializing data variable
    name = None
    fname = None
    dob = None
    pan = None
    nameline = []
    dobline = []
    panline = []
    text0 = []
    text1 = []
    text2 = []

    # Searching for PAN
    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = lin.replace('\n', '')
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)

    text1 = list(filter(None, text1))
    # print(text1)

    # to remove any text read from the image file which lies before the line 'Income Tax Department'

    lineno = 0  # to start from the first line of the text file.

    for wordline in text1:
        xx = wordline.split('\n')
        if ([w for w in xx if re.search(
                '(INCOME TAX DEPARTMENT % GOVT. OFINDIA|INCOMETAXDEPARTMENTLGOVT.OFINDIA|INCOMETAXDEPARWENT @|INCOME TAX DEPARTMENT|INCOME TAX DEPARTMENT  ¢|INCOMETAX DEPARTMENT gs GOV\'T OF Be|mcommx|INCOME|TAX|GOW|GOVT|GOVERNMENT|OVERNMENT|VERNMENT|DEPARTMENT|EPARTMENT|PARTMENT|ARTMENT|INDIA|NDIA)$',
                w)]):
            text1 = list(text1)
            lineno = text1.index(wordline)
            break

    # text1 = list(text1)
    text0 = text1[lineno + 1:]

    # print(text0)  # Contains all the relevant extracted text in form of a list - uncomment to check

    def findword(textlist, wordstring):
        lineno = -1
        for wordline in textlist:
            xx = wordline.split()
            if [w for w in xx if re.search(wordstring, w)]:
                lineno = textlist.index(wordline)
                textlist = textlist[lineno + 1:]
                return textlist
        return textlist

    remove_lower = lambda text: re.sub('[a-z]', '', text)
    try:

        # Cleaning first names, better accuracy
        name = text0[0]
        name = name.strip()
        name = name.replace("8", "B")
        name = name.replace("0", "D")
        name = name.replace("6", "G")
        name = name.replace("1", "I")
        name = name.replace("\'", "")
        name = re.sub('[^A-Z] +', ' ', name)
        result = remove_lower(name)
        name = result
        name = name.strip()

        # Cleaning Father's name
        fname = text0[1]
        fname = fname.strip()
        fname = fname.replace("8", "S")
        fname = fname.replace("0", "O")
        fname = fname.replace("6", "G")
        fname = fname.replace("1", "I")
        fname = fname.replace("\"", "A")
        fname = fname.replace("\'", "")
        fname = re.sub('[^A-Z] +', ' ', fname)
        result = remove_lower(fname)
        fname = result
        fname = fname.strip()

        # Cleaning DOB
        dob = text0[2]
        dob = dob.rstrip()
        dob = dob.lstrip()
        dob = dob.replace('l', '/')
        dob = dob.replace('L', '/')
        dob = dob.replace('I', '/')
        dob = dob.replace('i', '/')
        dob = dob.replace('|', '/')
        dob = dob.replace('\"', '/1')
        # dob = dob.replace("/", " ")
        remove_extra = lambda text: re.sub('[^0-9/]', '', text)
        dob = dob.lstrip()
        dob = dob.rstrip()
        result = remove_extra(dob)
        dob = result

        # Cleaning PAN Card details
        text0 = findword(text1, '(INCOME TAX DEPARTMENT % GOVT. OFINDIA|INCOMETAXDEPARTMENTLGOVT.OFINDIA|Pormanam|Number|umber|Account|ccount|count|Permanent|ermanent|manent|wumm)$')
        panline = text0[0]
        pan = panline.rstrip()
        pan = pan.lstrip()
        pan = pan.replace(" ", "")
        pan = pan.replace("\"", "")
        pan = pan.replace(";", "")
        pan = pan.replace("%", "L")
        remove_other = lambda text: re.sub('[^A-Z0-9]', '', text)
        result = remove_other(pan)
        if len(result) > 10:
            pan = result[:10]
        else:
            pan = result
    except:
        pass

    # Making tuples of data
    data = {'Name': name, 'Father Name': fname, 'Date of Birth': dob, 'PAN': pan}

    print(data)

    # Writing data into JSON
    try:
        to_unicode = unicode
    except NameError:
        to_unicode = str

    # Write JSON file
    with io.open('data.json', 'w', encoding='utf-8') as outfile:
        str_ = json.dumps(data, indent=4, sort_keys=True, separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

    # Read JSON file

    # Reading data back JSON(give correct path where JSON is stored)

    # pandas.read_json("data.json").to_excel("resultant_info.xlsx")

    # Opening JSON file and loading the data
    # into the variable data
    with open('data.json') as json_file:
        data = json.load(json_file)

    employee_data = data

    fields = employee_data
    l = list()
    l.append(fields)
    try:
        with open('data_file.csv', 'r') as csvfile:
            count = 1
    except:
        count = 0
    with open('data_file.csv', 'a') as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        if count == 0:
            writer.writeheader()

        # writing data rows
        writer.writerows(l)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to input image to be OCR'd")
    ap.add_argument("-p", "--preprocess", type=str, default="thresh",
                    help="type of preprocessing to be done, choose from blur, linear, cubic or bilateral")
    args1 = vars(ap.parse_args())
    ocr_apply(args1)
