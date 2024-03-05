# omr

### Credit to the original OMR code creator, rbaron (https://github.com/rbaron). Without his script, I could not accomplish this pilot project. Thank you, rbaron!

The basic data collection for operation using OMR was built on top of rbaron's code. The project mainly uses the OpenCV package and pandas DataFrame.

How to use:
1. Print the Sample_sheet.png in the sheet folder.
2. Fill in the employee ID with the bubble. For example: bubble 20 and bubble 5 make the ID 25.
3. Fill in the year bubble 24 as 2024, Month bubble Mar as March, Day bubble 20 + 7 means 27.
4. Fill in the Daily Activity column for the Activity ID combination.
5. Fill in the Item column for the Item processing difficulties level.
6. Fill in the Pieces columns for the number of pieces that you processed. The maximum number is 1999. 'T' stands for one thousand, 'H' stands for hundred, such as 2H means 200.
7. After you have filled the marks, scan and save them into the input folder.
8. Install required packages such as OpenCV, NumPy, and pandas.
9. Make sure that you have the same folder structure as my GitHub, which includes exports, img, and input folders.
10. Use either terminal, PyCharm, or any other Python interpreter to run the omr.py script.
11. The OMR recognized data frames will be exported as CSV to the exports folder. 

For more information, feel free to contact me via hank.lo.kyaw@gmail.com!

Enjoy!
