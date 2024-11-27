% reverse_words_in_file(InputFile, OutputFile)
% Главный предикат. Переворачивает все слова в строках файла, сохраняя их порядок, 
% и записывает результат в новый файл.
% InputFile - имя входного файла.
% OutputFile - имя выходного файла.
reverse_words_in_file(InputFile, OutputFile) :-
    open(InputFile, read, InStream),          % Открываем входной файл для чтения.
    open(OutputFile, write, OutStream),       % Открываем выходной файл для записи.
    process_lines(InStream, OutStream),       % Обрабатываем строки файла.
    close(InStream),                          % Закрываем входной поток.
    close(OutStream).                         % Закрываем выходной поток.

% process_lines(InStream, OutStream)
% Обрабатывает строки из входного потока и записывает результат в выходной поток.
process_lines(InStream, OutStream) :-
    read_line_to_string(InStream, Line),      % Считываем строку из входного файла.
    ( Line \= end_of_file ->                  % Если строка не конец файла:
        split_string(Line, " ", "", Words),   % Разделяем строку на список слов.
        maplist(reverse_word, Words, ReversedWords), % Переворачиваем каждое слово.
        atomic_list_concat(ReversedWords, " ", ReversedLine), % Соединяем перевёрнутые слова в строку.
        writeln(OutStream, ReversedLine),     % Записываем строку в выходной файл.
        process_lines(InStream, OutStream)    % Рекурсивно обрабатываем оставшиеся строки.
    ; true ).                                 % Если достигнут конец файла, завершаем.

% reverse_word(Word, Reversed)
% Переворачивает символы в слове.
% Word - слово для переворачивания.
% Reversed - перевёрнутое слово.
reverse_word(Word, Reversed) :-
    string_chars(Word, Chars),                % Преобразуем строку в список символов.
    reverse(Chars, ReversedChars),            % Переворачиваем список символов.
    string_chars(Reversed, ReversedChars).    % Преобразуем перевёрнутый список обратно в строку.
