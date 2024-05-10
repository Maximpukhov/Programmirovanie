#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double susceptible(double aI, double S, double I, double N, double aE, double E)
{
    return (-1 * ((aI * S * I) / N) + ((aE * S * E) / N));
}

double ezaraz(double aI, double S, double I, double N, double aE, double E, double K, double P)
{
    return ((((aI * S * I) / N) + ((aE * S * E) / N)) - ((K + P) * E));
}

double infected(double K, double E, double I, double m, double B)
{
    return (K * E - B * I - m * I);
}

double recovered(double B, double I, double P, double E)
{
    return (B * I + P * E);
}

double dead(double m, double I)
{
    return (m * I);
}

double group_SEIRD(double S_prev, double E_prev, double I_prev, double R_prev, double D_prev, double day)
{
    double aI = 0.999;
    double aE = 0.999;
    double K = 0.042;
    double P = 0.952;
    double m = 0.0188;
    double B = 0.999;

    double h = day/90;
    
    int N;
    for (int i = 0; i < day; i++)
    {
        double S2, E2, I2, R2, D2;

        double S1, E1, I1;

        N = S_prev + E_prev + I_prev + R_prev + D_prev;

        S1 = S_prev + (h * susceptible(aI, S_prev, I_prev, N, aE, E_prev));
        E1 = E_prev + (h * ezaraz(aI, S_prev, I_prev, N, aE, E_prev, K, P));
        I1 = I_prev + (h * infected(K, E_prev, I_prev, m, B));

        S2 = S_prev + (h / 2 * (susceptible(aI, S_prev, I_prev, N, aE, E_prev) + susceptible(aI, S1, I1, N, aE, E1)));
        E2 = E_prev + (h / 2 * (ezaraz(aI, S_prev, I_prev, N, aE, E_prev, K, P) + ezaraz(aI, S1, I1, N, aE, E1, K, P)));
        I2 = I_prev + (h / 2 * (infected(K, E_prev, I_prev, m, B) + infected(K, E1, I1, m, B)));
        R2 = R_prev + (h / 2 * (recovered(B, I_prev, P, E_prev) + recovered(B, I1, P, E1)));
        D2 = D_prev + (h / 2 * (dead(m, I_prev) + dead(m, I1)));

        S_prev = S2;
        E_prev = E2;
        I_prev = I2;
        R_prev = R2;
        D_prev = D2;
    }
    
    return (K * E_prev) / 0.58;
}

int main()
{

    double ezaraz = 99;
    double infected = 0;
    double recovered = 24;
    double dead = 0;
    double susceptible = 2798170 - ezaraz - recovered;

    FILE *fd = fopen("1.txt", "w");
    if (fd == NULL)
    {
        printf("Error open file\n");
        exit(EXIT_FAILURE);
    }

    char buf[100];
    for (int i = 1; i < 91; i++)
    {
        double Fk = group_SEIRD(susceptible, ezaraz, infected, recovered, dead, i);
        sprintf(buf, "%d %.6f\n", i, Fk);
        fwrite(buf, sizeof(char), strlen(buf), fd);
    }

    fclose(fd);
}