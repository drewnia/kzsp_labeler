Осноной запуск через 
python3 Tool_for_yolo.py

!!Работать на английской раскладке!!

Запустите Tool_for_yolo(у вас откроется маленькое черное окошко), нажмите клавишу "q" если у вас есть данные для проверки
если данных нет, то необходимо их создать для этого выполните следующую инструкцию:  
1. в ./NeuralNetworks/yolov5/images/work_dir загрузите неподготовленные фотографии одного сектора для разметки
2. переключитесь на меленькое окошко и нажмите на 'e' для подготовки изображений  
3. нажмите на клавишу 'e' для получения предсказаний нейронки  
4. когда предсказания будут готовы, появится окно с изображением  


Используя кнопки 'n' и 'n' выбирайте фотогрфию и нажмите 'y'


Инструкция для разметки:
`ЛКМ, ПКМ - левая кнопка и правая кнопка мыши соответственно`  
1. для перемещения прямоугольников: наведитесь на нужный прямоугольник, зажмите ПКМ и перемещайте, отпустите в нужном месте
2. для рисования: зажмите ЛКМ(сохранится начальная точка) и тяните для изменения размера, отпустите что бы оставить его на изображении  
3. для удаления: сместите нужный прямоугольник при помощи ПКМ и нажмите 'd'  
4. для сохранения всех прямоугольников нажмите 's'  
5. нажмите ESC для перехода к выбору фото  
6. 'r' - удалить все прямоугольники в изображении  

После разметки всех фото, размеченные фото и лейблы к ним будут находиться в ./NeuralNetworks/traindata/images и ./NeuralNetworks/traindata/labels соответственно.