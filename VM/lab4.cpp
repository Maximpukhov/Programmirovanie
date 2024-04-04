#include <stdio.h>

void setMatrix(double* str[], int size, char* filename)
{
    printf("%s\n",filename);
    FILE* matrix = fopen(filename, "r");
    double q;
    int c, g, i, j;
    char s;
    for(i = 0; i < 2; i++)
    {
        for(j = 0; j < size; j++)
        {
            q = 1.0;
            c = 0;
            g = 1;
            s = fgetc(matrix);
            do
            {
                if(s == '-') { c = 1; s = fgetc(matrix); continue; }
                if(s == '.') { g = 0; s = fgetc(matrix); continue; }
                if(g) str[i][j] = str[i][j]*10 + (s - 48);
                else {q *= 0.1; str[i][j] += q*(s - 48); }
                s = fgetc(matrix);
            }while(s != ' ' && s != '\n');
            if(c) str[i][j] *= -1;
        }
    }
    for(j = 0; j <= 1; j++)
    {
        q = 1.0;
        c = 0;
        g = 1;
        s = fgetc(matrix);
        do
        {
            if(s == '-') { c = 1; s = fgetc(matrix); continue; }
            if(s == '.') { g = 0; s = fgetc(matrix); continue; }
            if(g) str[2][j] = str[2][j]*10 + (s-48);
            else {q *= 0.1; str[2][j] += q*(s - 48); }
            s = fgetc(matrix);
        }while( s != '\n');
        if(c) str[2][j] *= -1;
    }
    
    fclose(matrix);
}

double Interpolation(double* str[], int size)
{
    double a, b, c = 0;
    int i, j;    
    for(i = 0; i < size; i++)
    {
        a = 1;  b = 1;
        for(j = 0; j < i; j++)
        {
            a *= (str[2][0] - str[0][j]);
            b *= (str[0][i] - str[0][j]);
        }
        for(j = i + 1; j < size; j++)
        {
            a *= (str[2][0] - str[0][j]);
            b *= (str[0][i] - str[0][j]);
        }
        c += ((str[1][i] *a) / b);
    }
    return c;
}
double Newtone_1(double* str[], int size)
{
    double q, c, s;
    double qwe[size][size] = {0};
    int i, j, a;

    for(i = 0; i < size; i++)
    {
        qwe[i][0] = str[1][i];
    }
    for(j = 1; j < size; j++)
    {
        for(i = 0; i < size - j; i++)
            qwe[i][j] = qwe[i + 1][j - 1] - qwe[i][j-1];
    }
    q = (str[2][0] - str[0][0])/(str[0][1] - str[0][0]);
    c = str[1][0];
    for(j = 1; j < size; j++)
    {
        a = 1;
        s = 1;
        for(i = 1; i <= j; i++)
        { a = a * i; s = s * (q - i + 1); }
        c = c + (qwe[0][j] * s)/ a;

    }

    return c;
}
int main(int a, char* filename[])
{
    double** str;
    double str_out, q;
    char s;
    int size = 0;
    int i,j;

    FILE* matrix = fopen(filename[1], "r");
    if(matrix == NULL)
    {
        printf("~The file %s not found or not connected~\n", filename[1]);
        return 0;
    }
    else
    {
        s = fgetc(matrix);
        while(s != '\n')
       {
            if(s == ' ')
                size++;
            s = fgetc(matrix);
        }
        size++;
        str = new double*[3];
        for(i = 0; i < 2; i++) { str[i] = new double[size]; }
        str[2] = new double[2];
        fclose(matrix);
    }
    
    setMatrix(str, size, filename[1]);
    printf("%d\n", size);
    for(i = 0; i < 2; i++)
    {
        for(j = 0; j < size; j++)
        {
            printf("%7.4f ",str[i][j]);
        }
        printf("\n");
    }
    printf("x = %f, out = %f\n", str[2][0], str[2][1]);
//intrepol
    str_out = Interpolation(str, size);

    if(str[2][1] > str_out)
        q = (str[2][1] - str_out)/str[2][1];
    else q = (str_out - str[2][1])/str[2][1];
    printf("str_out = %f, q = %f\n", str_out, q);
//Newtone formula
    str_out = Newtone_1(str, size);
    printf("Newtone formule 1 str_out = %f\n", str_out);

    for(i = 0; i < 3; i++)
    {
        delete(str[i]);
    }

    delete(str);
    return 0;
}
