# rubix-cube

This project involves implementation of encryption of image using modified-rubix cube algorithm.

1.Take an Image as an input(mxn).
2.Convert it into  Gray-scale image.
3.Take two variables m and n where, m=number of rows, n=number of columns.
4.For encryption two keys are required. The two keys Kr and Kc are generated using logistic equation.
5.Take a variable itrMax, which would give the total  number of  iteration, itrMax =(Kr[m/2] xor Kc[n/2])%5 +10.
6.Initialize a variable ITER=0, and then keep on incrementing the value of ITER by 1 upto itrmax after execution of all above the steps.
7.Image shifting row wise for each row of the image 
        (a).row i is left, or right, circular-shifted by Kr(i) positions (image pixels are moved Kr(i) positions to the left or right direction, and the first pixel moves in last pixel.), according to a randomized function.
8.Image shifting column wise, for each column of the image
        (a).column j is down, or up, circular-shifted by Kc(i) positions, according to randomized function.
9.Xoring of the image row wise,for each row of the image ,Xor the row with Kc.
10.Xoring of the image column wise,for each column of the image ,Xor the column with Kr.
11.If ITER=ITERMax then encrypted image is created and encryption process in done, else switch to 7th step.

TESTING : 
  The strength of the algorithm was evaluated using parameters like npcr,uaci,correlation coefficient.
 
In order to excute the application, run rubix-cube.py file.
