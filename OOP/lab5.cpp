#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <iomanip>
#include <math.h>

using namespace std;

class Figure
{
protected:
    int x;
    int y;

public:
    Figure()
    {
        srand(time(0));
        x = rand() % 100;
        y = rand() % 100;
    }
    Figure(int x, int y)
    {
        this->x = x;
        this->y = y;
    }
    void input()
    {
        cout << "Enter x and y: ";
        cin >> x >> y;
    }
    void output()
    {
        cout << "x = " << x << "y = " << y << endl;
    }
    ~Figure()
    {
    }
};

class Line : public Figure
{
protected:
    double len;
    int x2;
    int y2;
    void calclen()
    {
        len = sqrt(pow(x2 - x, 2) + pow(y2 - y, 2));
    }

public:
    Line() : Figure()
    {
        x2 = rand() % 100;
        y2 = rand() % 100;
        calclen();
    }
    Line(int x, int y, int x2, int y2) : Figure(x, y)
    {
        this->x2 = x2;
        this->y2 = y2;
        calclen();
    }
    void input()
    {
        Figure::input();
        cout << "Enter x2 and y2: ";
        cin >> x2 >> y2;
        calclen();
    }
    void output()
    {
        Figure::output();
        cout << "x2 = " << x2 << " y2 = " << y2 << endl;
        cout << "Len line =  " << len << endl;
    }
    double getLength()
    {
        return len;
    }
    ~Line() {}
};

class Square : public Line
{
protected:
    int sidelen;

public:
    Square() : Line()
    {
        sidelen = rand() % 100;
    }
    Square(int x1, int y1, int x2, int y2) : Line(x1, y1, x2, y2)
    {
        sidelen = (sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2)));
    }
    void input()
    {
        Line::input();
        cout << "Enter side square: ";
        cin >> sidelen;
    }

    void output()
    {
        Line::output();
        cout << "Side square = " << sidelen << endl;
    }
    double getArea()
    {
        return sidelen * sidelen;
    }

    double getPerimeter()
    {
        return 4 * sidelen;
    }
    void print()
    {
        cout << sidelen << endl;
    }
    ~Square() {}
};

class Rectangle : public Square
{
private:
    int width;
    int height;
    double area;
    double per;

public:
    Rectangle() : Square()
    {
        width = rand() % 100;
        height = rand() % 100;
    }
    Rectangle(int x1, int y1, int x2, int y2) : Square(x1, y1, x2, y2)
    {
        this->width = abs(x2 - x1);
        this->height = abs(y2 - y1);
        if (width == height)
        {
            cout << "don`t rectangle" << endl;
            exit(EXIT_FAILURE);
        }
    }
    void print()
    {
        cout << width << endl;
    }
    void input()
    {
        Square::input();
        cout << "Enter width and height rectangle: ";
        cin >> width >> height;
    }
    void output()
    {
        Square::output();
        cout << "Width rectangle = " << width << endl;
        cout << "Height rectangle = " << height << endl;
    }

    double getArea()
    {
        return width * height;
    }

    double getPerimeter()
    {
        return 2 * (width + height);
    }
    ~Rectangle() {}
};
class Circle : public Figure
{
private:
    double radius;

public:
    Circle() : Figure()
    {
        radius = rand() % 100;
    }

    Circle(int x, int y, double radius) : Figure(x, y)
    {
        this->radius = radius;
    }

    double getRadius() const
    {
        return radius;
    }
    double getPer() const
    {
        return 2 * 3.14159 * radius;
    }
};

class Ellipse : public Circle
{
private:
    double minorRadius;

public:
    Ellipse() : Circle()
    {
        minorRadius = rand() % 100;
    }

    Ellipse(int x, int y, double majorRadius, double minorRadius) : Circle(x, y, majorRadius)
    {
        this->minorRadius = minorRadius;
    }

    double getMinorRadius() const
    {
        return minorRadius;
    }

    double getMajorRadius() const
    {
        return getRadius();
    }

    double getArea() const
    {
        return 3.14159 * getMajorRadius() * minorRadius;
    }

    double getPerimeter() const
    {
        double h = ((getMajorRadius() - minorRadius) * (getMajorRadius() - minorRadius)) / ((getMajorRadius() + minorRadius) * (getMajorRadius() + minorRadius));
        return 3.14159 * (getMajorRadius() + minorRadius) * (1 + (3 * h) / (10 + sqrt(4 - 3 * h)));
    }
};
int main()
{
    Line line1(2, 4, 7, 3);
    Line line2;
    Square square1(2,2, 4,2);
    Square square2;
    Rectangle rectangle1(2, 2, 4, 1);
    Rectangle rectangle2;

    cout << "Len Line 1 =  " << line1.getLength() << endl;
    cout << "Len line 2 =  " << line2.getLength() << endl;

    cout << "Per square 1 = " << square1.getPerimeter() << endl;
    cout << "Area square 1 = " << square1.getArea() << endl;
    cout << "Per square 2 = " << square2.getPerimeter() << endl;
    cout << "Area square 2 = " << square2.getArea() << endl;

    cout << "Per rectangle 1 = " << rectangle1.getPerimeter() << endl;
    cout << "Area rectangle 1 = " << rectangle1.getArea() << endl;
    cout << "Per rectangle 2 = " << rectangle2.getPerimeter() << endl;
    cout << "Area rectangle 2 = " << rectangle2.getArea() << endl;

    Rectangle *arr = new Rectangle[3];
    arr[0] = Rectangle(2, 3, 8, 5);
    arr[1] = Rectangle(9, 7, 3, 8);
    arr[2] = Rectangle(12, 6, 2, 3);
    for (int i = 0; i < 3; i++)
    {
        cout << "Per rectangle " << i + 1 << " = " << arr[i].getPerimeter() << endl;
        cout << "Area rectangle " << i + 1 << " = " << arr[i].getArea() << endl;
    }
    delete[] arr;
}
