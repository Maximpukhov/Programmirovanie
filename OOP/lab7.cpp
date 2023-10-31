#include <iostream>

using namespace std;

template <typename D>
class Node
{
public:
    Node *pnext;
    D value;
    Node(D value = D(), Node *pnext = NULL)
    {
        this->value = value;
        this->pnext = pnext;
    }
};
template <typename T>
class List : public Node<T>
{
public:
    List() : head(NULL), Node<T>(0, head)
    {
        this->count = 0;
    }

    int get_count()
    {
        return count;
    }

    void delete_start()
    {
        Node<T> *temp = head;
        head = head->pnext;
        delete temp;
        count--;
    }

    void delete_node(int ctr)
    {
        if (ctr == 0)
        {
            delete_start();
        }
        else
        {
            Node<T> *temp = this->head;  // delete node
            Node<T> *temp1 = this->head; // temp node
            for (int i = 0; i < ctr; i++)
            {
                temp = temp->pnext;
            }
            for (int i = 0; i < ctr - 1; i++)
            {
                temp1 = temp1->pnext;
            }
            temp1->pnext = temp->pnext;
            delete temp;
            count--;
        }
    }

    T lookup(int ctr)
    {
        int cou = 0;
        Node<T> *cur = this->head;
        for (; cur != NULL; cur = cur->pnext, cou++)
        {
            if (ctr == cou)
            {
                return cur->value;
            }
        }
        return -1;
    }

    void add_end(T value)
    {
        try
        {
            if (head == NULL)
            {
                head = new Node<T>(value);
            }
            else
            {
                Node<T> *cur = this->head;
                while (cur->pnext != NULL)
                {
                    cur = cur->pnext;
                }
                cur->pnext = new Node<T>(value);
            }
            count++;
        }
        catch (const std::bad_alloc &e)
        {
            cerr << "Memory allocation error: " << e.what() << endl;
            throw;
        }
    }

    virtual void add_start(T value)
    {
        head = new Node<T>(value, head);
        count++;
    }

    void insert(T value, int ind)
    {
        if (ind == 0)
        {
            add_start(value);
        }
        else
        {
            Node<T> *temp = this->head;
            for (int i = 0; i < ind - 1; i++)
            {
                temp = temp->pnext;
            }
            Node<T> *nodee = new Node<T>(value, temp->pnext);
            temp->pnext = nodee;
            count++;
        }
    }

    virtual ~List()
    {
        while (count)
        {
            Node<T> *temp = head;
            head = head->pnext;
            delete temp;
            count--;
        }
    }

protected:
    int count;
    Node<T> *head;
};

template <typename T>
class Stack : public List<T>
{
private:
    Node<T> *top;
    int size;

public:
    Stack() : top(NULL), size(0)
    {
    }

    void add_start(int value)
    {
        top = new Node<T>(value, top);
        size++;
    }

    T pop()
    {
        Node<T> *next;
        int num;
        if (top == NULL)
        {
            throw std::underflow_error("Stack underflow");
            return -1;
        }
        next = top->pnext;
        num = top->value;
        delete top;
        top = next;
        size--;
        return num;
    }

    T lookup(int size)
    {
        int cou = 0;
        Node<T> *cur = this->top;
        for (; cur != NULL; cur = cur->pnext, cou++)
        {
            if (size == cou)
            {
                return cur->value;
            }
        }
        return -1;
    }

    virtual ~Stack()
    {
        while (size > 0)
        {
            Node<T> *next = top;
            next = top->pnext;
            delete next;
            size--;
        }
    }
};
template <typename T>
class Queue : public List<T>
{
private:
    Node<T> *head;
    Node<T> *tail;
    int size;

public:
    Queue() : head(NULL), tail(NULL), size(0)
    {
    }

    void enqueue(int value)
    {
        Node<T> *oldtail = tail;
        tail = new Node<T>(value);
        if (head == NULL)
        {
            head = tail;
        }
        else
        {
            oldtail->pnext = tail;
        }
        size++;
    }

    int getSize() { return size; }

    T dequeue()
    {
        int num;
        Node<T> *p;
        if (size == 0)
        {
            throw std::underflow_error("Queue underflow");
        }
        num = head->value;
        p = head->pnext;
        delete head;
        head = p;
        size--;
        return num;
    }

    ~Queue()
    {
        while (size > 0)
        {
            Node<T> *p = head;
            p = head->pnext;
            delete p;
            size--;
        }
    }
};
int main()
{
    cout << "List" << endl;

    List<int> obj1;
    obj1.add_end(5);
    obj1.add_end(11);
    obj1.insert(25, 1);
    obj1.add_end(30);
    obj1.add_end(48);
    obj1.add_end(97);

    for (int i = 0; i < obj1.get_count(); i++)
    {
        cout << obj1.lookup(i) << endl;
    }

    obj1.delete_node(0);
    obj1.delete_node(3);

    cout << "delete" << endl;
    for (int i = 0; i < obj1.get_count(); i++)
    {
        cout << obj1.lookup(i) << endl;
    }

    cout << "Stack" << endl;
    int i, value;
    Stack<int> object2;

    for (i = 1; i <= 10; i++)
        object2.add_start(i);

    try
    {
        cout << object2.lookup(5) << endl;

        for (i = 1; i < 11; i++) // i
        {
            value = object2.pop();
            cout << value << endl;
        }
    }
    catch (const std::underflow_error &err)
    {
        cerr << "Error: " << err.what() << endl;
    }

    cout << "Queue" << endl;
    Queue<int> object3;

    int s;
    int num;

    for (s = 1; s <= 10; s++)
    {
        object3.enqueue(s);
    }

    try
    {
        for (s = 1; s < 11; s++) // s
        {
            num = object3.dequeue();
            cout << "elem: " << num << endl;
        }
    }
    catch (const std::underflow_error &err)
    {
        cerr << "Error: " << err.what() << endl;
    }

    cout << "size = " << object3.getSize() << endl;
    return 0;
}
