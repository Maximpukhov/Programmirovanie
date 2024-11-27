% Делаем предикат flight/4 динамическим, чтобы можно было добавлять, удалять и изменять записи в базе данных.
:- dynamic flight/4.

% === Загрузка базы данных ===
% Загружает базу данных из указанного файла.
% Если файл существует, он загружается с помощью consult/1.
% Если файла нет, ничего не делается (true).
load_database(File) :-
    exists_file(File) ->               % Проверяем, существует ли файл.
    consult(File);                     % Если файл существует, загружаем его.
    true.                              % Если файл не найден, просто продолжаем.

% === Сохранение базы данных ===
% Сохраняет текущее состояние базы данных в указанный файл.
save_database(File) :-
    tell(File),                        % Открываем файл для записи.
    listing(flight),                   % Выводим все факты flight/4 в файл.
    told.                              % Закрываем файл.

% === Показать все записи ===
% Выводит все рейсы в базе данных.
show_flights :-
    forall(flight(Num, Dest, Time, Price),    % Проходим по всем фактам flight/4.
           format("~w ~w ~w ~w~n", [Num, Dest, Time, Price])).  % Выводим каждый рейс.

% === Добавление рейса ===
% Позволяет пользователю добавлять новые рейсы в базу данных.
add_flight :-
    write("Введите данные рейса (номер, пункт назначения, время отправления, стоимость билета):"), nl,
    read(Num), read(Dest), read(Time), read(Price),  % Считываем данные рейса.
    assertz(flight(Num, Dest, Time, Price)),         % Добавляем рейс в базу.
    write("Добавить еще запись? (yes/no): "), nl,    % Спрашиваем, нужно ли добавить ещё.
    read(Choice),                                    % Считываем выбор пользователя.
    ( Choice == yes -> add_flight                   % Если "yes", снова вызываем add_flight.
    ; true ).                                        % Иначе завершаем.

% === Удаление рейса ===
% Позволяет пользователю удалять рейсы из базы данных.
delete_flight :-
    write("Введите номер рейса для удаления: "), nl,
    read(Num),                               % Считываем номер рейса.
    retractall(flight(Num, _, _, _)),        % Удаляем все факты с этим номером рейса.
    write("Удалить еще запись? (yes/no): "), nl,  % Спрашиваем, нужно ли удалить ещё.
    read(Choice),                                 % Считываем выбор пользователя.
    ( Choice == yes -> delete_flight            % Если "yes", снова вызываем delete_flight.
    ; true ).                                   % Иначе завершаем.

% === Запрос рейсов ===
% Находит рейсы по заданному пункту назначения и времени.
query_flights :-
    write("Введите пункт назначения: "), nl,
    read(Dest),                          % Считываем пункт назначения.
    write("Введите текущее время (часы:минуты): "), nl,
    read(CurrentTime),                   % Считываем текущее время.
    findall((Num, Dest, DepTime, Price), % Собираем все рейсы, соответствующие условиям:
            (flight(Num, Dest, DepTime, Price), 
             within_next_six_hours(CurrentTime, DepTime)), 
            Flights),
    ( Flights = [] ->                    % Если список рейсов пуст:
        write("Рейсы не найдены.")       % Сообщаем, что рейсов нет.
    ; find_min_price(Flights, MinPrice), % Иначе находим минимальную цену.
      include(has_price(MinPrice), Flights, CheapestFlights), % Отбираем рейсы с минимальной ценой.
      write("Рейсы с минимальной ценой: "), nl,
      forall(member((N, D, T, P), CheapestFlights),          % Выводим рейсы с минимальной ценой.
             format("~w ~w ~w ~w~n", [N, D, T, P])) ).

% === Проверка рейсов в пределах 6 часов ===
% Определяет, находится ли время отправления в пределах следующих 6 часов.
within_next_six_hours(CurrentTime, DepartureTime) :-
    parse_time(CurrentTime, CurrHours, CurrMinutes),   % Разбираем текущее время.
    parse_time(DepartureTime, DepHours, DepMinutes),   % Разбираем время отправления.
    CurrTotal is CurrHours * 60 + CurrMinutes,         % Преобразуем текущее время в минуты.
    DepTotal is DepHours * 60 + DepMinutes,            % Преобразуем время отправления в минуты.
    ( DepTotal > CurrTotal ->                          % Если отправление в тот же день:
      Diff is DepTotal - CurrTotal
    ; Diff is 1440 - CurrTotal + DepTotal ),           % Если отправление на следующий день.
    Diff =< 360.                                       % Проверяем, не превышает ли разница 6 часов.

% === Найти минимальную цену ===
% Находит минимальную цену среди рейсов.
find_min_price(Flights, MinPrice) :-
    findall(Price, member((_, _, _, Price), Flights), Prices), % Собираем все цены.
    min_list(Prices, MinPrice).                                % Находим минимальное значение.

% === Фильтр по цене ===
% Проверяет, имеет ли рейс заданную цену.
has_price(MinPrice, (_, _, _, Price)) :-
    Price =:= MinPrice.

% === Разбор времени ===
% Разбирает строку времени "часы:минуты" в числовые значения.
parse_time(Time, Hours, Minutes) :-
    split_string(Time, ":", "", [H, M]),     % Разделяем строку на часы и минуты.
    number_string(Hours, H),                 % Преобразуем часы в число.
    number_string(Minutes, M).               % Преобразуем минуты в число.

% === Меню программы ===
% Отображает меню программы и обрабатывает выбор пользователя.
menu :-
    write("1. Показать все рейсы"), nl,
    write("2. Добавить рейс"), nl,
    write("3. Удалить рейс"), nl,
    write("4. Найти рейсы"), nl,
    write("5. Выйти"), nl,
    write("Выберите опцию: "), nl,
    read(Choice),                           % Считываем выбор пользователя.
    ( Choice == 1 -> show_flights, menu;    % Вывод рейсов.
      Choice == 2 -> add_flight, menu;      % Добавление рейса.
      Choice == 3 -> delete_flight, menu;   % Удаление рейса.
      Choice == 4 -> query_flights, menu;   % Поиск рейсов.
      Choice == 5 -> save_database('flights_db.pl');  % Сохранение и выход.
      write("Неверный выбор!"), nl, menu ). % Неверный ввод, повторяем меню.
