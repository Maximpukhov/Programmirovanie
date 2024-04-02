#include <stdio.h>
#include <math.h>

double fun(float x)
{
    return 1 - 0.5 * x*x * log(x) + 0.3 * sqrt(x);
}

double hord(double a, double b, double p)
{
    double c = 1, f1, f2, f3;

    do
    {
        f1 = fun(b);
        f2 = fun(a);
        c = b - f1 * (b - a)/(f1-f2);



        f3 = fun(c);
        if(signbit(f2) != signbit(f3))
        {
            b = c;
        }
        if(signbit(f1) != signbit(f3))
        {
            a = c;
        }
//        break;
    } while (fabs(fabs(a) - fabs(b)) < p);
    return c;
}

int main()
{
    double out, a, b, p;
    a = 1.5, b = 2.5, p = 0.000001;
    printf("%f %f %f\n", a, b, p);
    if(a <= 0)
    {
        printf("no licvid\n");
        return 0;
    }
    out = hord(a, b, p);
    printf("%f\n", out);
    return 0;
}
