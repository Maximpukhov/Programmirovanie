% square_third(List, Result)
% Главный предикат. Возводит в квадрат каждый третий элемент списка.
% List - входной список.
% Result - выходной список с изменёнными элементами.
square_third(List, Result) :-
    square_third(List, 1, Result).  % Начинаем обработку с индекса 1.

% Базовый случай рекурсии: если список пуст, результат также пуст.
square_third([], _, []).

% Рекурсивный случай: обрабатываем каждый третий элемент.
% H - текущий элемент списка, T - оставшаяся часть списка.
% Index - текущий индекс элемента.
% Если индекс делится на 3 без остатка, возводим элемент в квадрат.
square_third([H|T], Index, [H2|Result]) :-
    0 is Index mod 3,                % Проверяем, является ли элемент третьим.
    H2 is H * H,                     % Возводим текущий элемент в квадрат.
    NextIndex is Index + 1,          % Увеличиваем индекс.
    square_third(T, NextIndex, Result).  % Продолжаем обработку оставшейся части списка.

% Рекурсивный случай: оставляем элемент без изменений.
% Если индекс не делится на 3, добавляем элемент в результат как есть.
square_third([H|T], Index, [H|Result]) :-
    0 =\= Index mod 3,               % Проверяем, что элемент не третий.
    NextIndex is Index + 1,          % Увеличиваем индекс.
    square_third(T, NextIndex, Result).  % Продолжаем обработку оставшейся части списка.