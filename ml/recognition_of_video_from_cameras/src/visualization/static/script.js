var color_choices = [
    "#FF00FF",
    "#8622FF",
    "#FE0056",
    "#00FFCE",
    "#FF8000",
    "#00B7EB",
    "#FFFF00",
    "#0E7AFE",
    "#FFABAB",
    "#0000FF",
    "#CCCCCC",
];

var radiansPer45Degrees = Math.PI / 4;

var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
ctx.lineJoin = 'bevel';

var img = new Image();
var rgb_color = color_choices[Math.floor(Math.random() * color_choices.length)]
var opaque_color =  'rgba(0,0,0,0.5)';

var scaleFactor = 1;
var scaleSpeed = 0.01;

var points = [];
var regions = [];
var masterPoints = [];
var masterColors = [];
let masterShapes = []

var drawMode
setDrawMode('polygon')
var constrainAngles = false;
var showNormalized = false;

var modeMessage = document.querySelector('#mode');
var coords = document.querySelector('#coords');

let filesArr = []

window.addEventListener("load", async (event) => {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    const firstImageData = await fetch('./getfile').then(res => res.json()).then(files => files[0]);
    if (firstImageData) {
        img.src = `data:image;base64,${firstImageData.file}`; // Установите источник изображения

        img.onload = function() {
            scaleFactor = 0.5; // Установите желаемый коэффициент масштаба
            canvas.style.width = img.width * scaleFactor + 'px';
            canvas.style.height = img.height * scaleFactor + 'px';
            canvas.width = img.width;
            canvas.height = img.height;
            ctx.drawImage(img, 0, 0); // Нарисуйте изображение на холсте
        };
    } else {
        console.error("Нет доступных изображений.");
        // Опционально обработайте случай, если изображения не найдены
    }
});

async function clipboard(selector) {
    // Сохранение координат
    await fetch("./saveCoords", {
        method: 'POST',
        body: JSON.stringify(document.querySelector('#python').value.replace(/\n/g, '')),
        headers: {
            'Content-Type': 'application/json' // Указываем тип контента
        }
    });

    // Получаем данные из формы
    const modelPath = document.querySelector('#model_path').value;
    const targetVideo = document.querySelector('#target_video').checked;
    const showVideo = document.querySelector('#show_video').checked;
    const confidence = document.querySelector('#confidence').value;
    const iou = document.querySelector('#iou').value;

    // Сохранение изображения с холста
    const dataURL = canvas.toDataURL('image/png'); // Получаем данные изображения в формате PNG

    try {
        // Сохранение изображения
        const response = await fetch('/save_image', {
            method: 'POST',
            body: JSON.stringify({
                image: dataURL
            }),
            headers: {
                'Content-Type': 'application/json' // Указываем тип контента
            }
        });

        if (!response.ok) {
            throw new Error('Ошибка при сохранении изображения');
        }

        // Получаем имя файла из заголовка Content-Disposition
        const disposition = response.headers.get('Content-Disposition');
        let filename = 'saved_image.png'; // Имя файла по умолчанию
        if (disposition && disposition.includes('filename=')) {
            filename = disposition.split('filename=')[1].replace(/"/g, ''); // Извлекаем имя файла
            filename = decodeURIComponent(filename); // Декодируем имя файла
        }

        // Автоматическая загрузка изображения
        const blob = await response.blob(); // Получаем Blob из ответа
        const url = URL.createObjectURL(blob); // Создаем URL для Blob
        const a = document.createElement('a'); // Создаем элемент ссылки
        a.href = url;
        a.download = filename; // Устанавливаем имя файла для сохранения
        document.body.appendChild(a); // Добавляем ссылку на страницу
        a.click(); // Симулируем клик для загрузки
        document.body.removeChild(a); // Удаляем ссылку после загрузки
        URL.revokeObjectURL(url); // Освобождаем память
        console.log('Image saved successfully and is being downloaded.');

        // Сохранение настроек
        const settingsResponse = await fetch('/save_settings', {
            method: 'POST',
            body: JSON.stringify({
                model_path: modelPath,
                target_video: targetVideo,
                show_video: showVideo,
                confidence: confidence,
                iou: iou
            }),
            headers: {
                'Content-Type': 'application/json' // Указываем тип контента
            }
        });

        if (!settingsResponse.ok) {
            throw new Error('Ошибка при сохранении настроек');
        }

        window.location.href = '/'; // Перенаправляем на главную страницу
    } catch (error) {
        console.error('Ошибка:', error); // Логируем ошибки
        alert(error.message); // Отображаем сообщение об ошибке
    }
}




function updatePythonOutput() {
    var code_template = JSON.stringify(masterShapes); // Преобразуем объекты в строку
    document.querySelector('#python').innerHTML = code_template; // Выводим в HTML
}

function zoom(clicks) {
    // if w > 60em, stop
    if ((scaleFactor + clicks * scaleSpeed) * img.width > 40 * 16) {
        return;
    }
    scaleFactor += clicks * scaleSpeed;
    scaleFactor = Math.max(0.1, Math.min(scaleFactor, 0.8));
    var w = img.width * scaleFactor;
    var h = img.height * scaleFactor;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
}

function closePath() {
    let name = prompt("Введите имя объекта:")
    if (name && points.length > 0) {
        masterShapes.push({
            name: name, // Сохраняем имя объекта
            points: points.slice() // последняя добавленная коллекция точек
        })
    }

    canvas.style.cursor = 'default';
    masterPoints.push(points);
    masterColors.push(rgb_color);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0);

    drawAllPolygons();
    points = [];

    // dont choose a color that has already been chosen
    var remaining_choices = color_choices.filter(function(x) {
        return !masterColors.includes(x);
    });

    if (remaining_choices.length == 0) {
        remaining_choices = color_choices;
    }

    rgb_color = remaining_choices[Math.floor(Math.random() * remaining_choices.length)];
    updatePythonOutput()
}

// placeholder image
img.src = 'https://assets.website-files.com/5f6bc60e665f54545a1e52a5/63d3f236a6f0dae14cdf0063_drag-image-here.png';
img.onload = function() {
    scaleFactor = 0.69;
    canvas.style.width = img.width * scaleFactor + 'px';
    canvas.style.height = img.height * scaleFactor + 'px';
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
};

function canvas_arrow(context, fromx, fromy, tox, toy, text, line) {
    ctx.fillStyle = 'white';
  let headlen = 15; // length of head in pixels
  let dx = tox - fromx;
  let dy = toy - fromy;
  const length = Math.sqrt(dx * dx + dy * dy);

  const normX = dx / length;
  const normY = dy / length;

  let perpX = -normY * 60;
  const perpY = normX * 60;

  const arrowStartX = (fromx + tox)/2;
  const arrowStartY = (fromy + toy)/2;

  let angle = Math.atan2(perpY, perpX);

  context.font = "45px serif";

  if (line === 1) {
    if(text === 'out'){
        context.moveTo(arrowStartX, arrowStartY);
        context.lineTo(arrowStartX + perpX, arrowStartY + perpY);
        context.lineTo(arrowStartX + perpX - headlen * Math.cos(angle - Math.PI / 6), arrowStartY + perpY - headlen * Math.sin(angle - Math.PI / 6));
        context.moveTo(arrowStartX + perpX, arrowStartY + perpY);
        context.lineTo(arrowStartX + perpX - headlen * Math.cos(angle + Math.PI / 6), arrowStartY + perpY - headlen * Math.sin(angle + Math.PI / 6));

        let ox = arrowStartX + perpX - 10
        let oy = arrowStartY + perpY

        if (fromy === toy && fromx > tox || fromy < toy && fromx === tox || fromy > toy && fromx > tox || fromy < toy && fromx > tox){
            oy -= 25
        } else if (fromy < toy && fromx < tox || fromy > toy && fromx < tox || fromy === toy && fromx < tox || fromy > toy && fromx === tox){
            oy += 15
        }
        context.textBaseline = 'top'
        context.fillStyle = '#000'
        let width = context.measureText(text).width + 10
        context.fillRect(ox-5, oy, width, parseInt("42px serif", 10));
        context.fillStyle = 'white'
        context.fillText(text, ox, oy);
    } else {
        context.moveTo(arrowStartX, arrowStartY);
        context.lineTo(arrowStartX - perpX, arrowStartY - perpY);
        context.moveTo(arrowStartX, arrowStartY);
        context.lineTo(arrowStartX - headlen * Math.cos(angle - Math.PI / 6), arrowStartY - headlen * Math.sin(angle - Math.PI / 6));
        context.moveTo(arrowStartX, arrowStartY);
        context.lineTo(arrowStartX - headlen * Math.cos(angle + Math.PI / 6), arrowStartY - headlen * Math.sin(angle + Math.PI / 6));

        let ox = arrowStartX - perpX - 10
        let oy = arrowStartY - perpY

        if (fromy === toy && fromx > tox || fromy < toy && fromx === tox || fromy > toy && fromx > tox || fromy < toy && fromx > tox){
            oy += 15
        } else if (fromy < toy && fromx < tox || fromy > toy && fromx < tox || fromy === toy && fromx < tox || fromy > toy && fromx === tox){
            oy -= 25
        }
        context.textBaseline = 'top'
        context.fillStyle = '#000'
        let width = context.measureText(text).width + 10
        context.fillRect(ox-5, oy, width, parseInt("45px serif", 10));
        context.fillStyle = 'white'
        context.fillText(text, ox, oy);
    }
  } else {
      if(text === 'out'){
        context.moveTo(arrowStartX, arrowStartY);
        context.lineTo(arrowStartX - perpX, arrowStartY - perpY);
        context.lineTo(arrowStartX - perpX + headlen * Math.cos(angle - Math.PI / 6), arrowStartY - perpY + headlen * Math.sin(angle - Math.PI / 6));
        context.moveTo(arrowStartX - perpX, arrowStartY - perpY);
        context.lineTo(arrowStartX - perpX + headlen * Math.cos(angle + Math.PI / 6), arrowStartY - perpY + headlen * Math.sin(angle + Math.PI / 6));

        let ox = arrowStartX - perpX - 10
        let oy = arrowStartY - perpY

        if (fromy === toy && fromx > tox || fromy < toy && fromx === tox || fromy > toy && fromx > tox || fromy < toy && fromx > tox){
            oy += 15
        } else if (fromy < toy && fromx < tox || fromy > toy && fromx < tox || fromy === toy && fromx < tox || fromy > toy && fromx === tox){
            oy -= 25
        }
        context.textBaseline = 'top'
        context.fillStyle = '#000'
        let width = context.measureText(text).width + 10
        context.fillRect(ox-5, oy, width, parseInt("45px serif", 10));
        context.fillStyle = 'white'
        context.fillText(text, ox, oy);
    } else {
        context.moveTo(arrowStartX, arrowStartY);
        context.lineTo(arrowStartX + perpX, arrowStartY + perpY);
        context.moveTo(arrowStartX, arrowStartY);
        context.lineTo(arrowStartX + headlen * Math.cos(angle - Math.PI / 6), arrowStartY + headlen * Math.sin(angle - Math.PI / 6));
        context.moveTo(arrowStartX, arrowStartY);
        context.lineTo(arrowStartX + headlen * Math.cos(angle + Math.PI / 6), arrowStartY + headlen * Math.sin(angle + Math.PI / 6));

        let ox = arrowStartX + perpX - 10
        let oy = arrowStartY + perpY

        if (fromy === toy && fromx > tox || fromy < toy && fromx === tox || fromy > toy && fromx > tox || fromy < toy && fromx > tox){
            oy -= 25
        } else if (fromy < toy && fromx < tox || fromy > toy && fromx < tox || fromy === toy && fromx < tox || fromy > toy && fromx === tox){
            oy += 15
        }
        context.textBaseline = 'top'
        context.fillStyle = '#000'
        let width = context.measureText(text).width + 10
        context.fillRect(ox-5, oy, width, parseInt("45px serif", 10));
        context.fillStyle = 'white'
        context.fillText(text, ox, oy);
    }
  }
}

function drawLine(x1, y1, x2, y2) {
    ctx.beginPath();
    // set widht
    ctx.lineWidth = 5;
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    /*ctx.beginPath();
    canvas_arrow(ctx, x1, y1, x2, y2, 'out');
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.beginPath();
    canvas_arrow(ctx, x1, y1, x2, y2, 'in');
    ctx.lineWidth = 2;
    ctx.stroke();*/
}

function getScaledCoords(e) {
    var rect = canvas.getBoundingClientRect();
    var x = e.clientX - rect.left;
    var y = e.clientY - rect.top;
    return [x / scaleFactor, y / scaleFactor];
}

function drawAllPolygons () {

    // draw all points for previous regions
    for (var i = 0; i < masterPoints.length; i++) {
        var newpoints = masterPoints[i];

        // set color
        ctx.strokeStyle = masterColors[i];
        for (var j = 1; j < newpoints.length; j++) {
            // draw all lines
            drawLine(newpoints[j - 1][0], newpoints[j - 1][1], newpoints[j][0], newpoints[j][1]);
            // fill
            ctx.beginPath();
            ctx.fillStyle = opaque_color;
            ctx.lineWidth = 5;
            ctx.moveTo(newpoints[0][0], newpoints[0][1]);
            for (var j = 1; j < newpoints.length; j++) {
                ctx.lineTo(newpoints[j][0], newpoints[j][1]);
            }
            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            for (var j = 1; j < newpoints.length; j+=2) {
                if(newpoints.length <= 2){
                    if (newpoints[j - 1][0] < newpoints[j][0]) {
                    ctx.beginPath();
                    canvas_arrow(ctx, newpoints[j - 1][0], newpoints[j - 1][1], newpoints[j][0], newpoints[j][1], 'out', 1);
                    ctx.lineWidth = 5;
                    ctx.fillStyle = 'white';
                    ctx.closePath();
                    ctx.stroke();
                    ctx.beginPath();
                    canvas_arrow(ctx, newpoints[j - 1][0], newpoints[j - 1][1], newpoints[j][0], newpoints[j][1], 'in', 1);
                    ctx.lineWidth = 5;
                    ctx.fillStyle = 'white';
                    ctx.closePath();
                    ctx.stroke();
                } else {
                    ctx.beginPath();
                    canvas_arrow(ctx, newpoints[j][0], newpoints[j][1], newpoints[j - 1][0], newpoints[j - 1][1], 'out', 0);
                    ctx.lineWidth = 5;
                    ctx.fillStyle = 'white';
                    ctx.closePath();
                    ctx.stroke();
                    ctx.beginPath();
                    canvas_arrow(ctx, newpoints[j][0], newpoints[j][1], newpoints[j - 1][0], newpoints[j - 1][1], 'in', 0);
                    ctx.lineWidth = 5;
                    ctx.fillStyle = 'white';
                    ctx.closePath();
                    ctx.stroke();
                }
                }
            }
        }
        drawLine(newpoints[newpoints.length - 1][0], newpoints[newpoints.length - 1][1], newpoints[0][0], newpoints[0][1]);
        // draw arc around each point
        let x = 0
        let y = 0
        for (var j = 0; j < newpoints.length; j++) {
            x += newpoints[j][0]
            y += newpoints[j][1]
            ctx.beginPath();
            ctx.strokeStyle = masterColors[i];
            ctx.arc(newpoints[j][0], newpoints[j][1], 5, 0, 2 * Math.PI);
            // fill with white
            ctx.fillStyle = 'white';
            ctx.fill();
            ctx.stroke();
        }
        //draw polygon name
        x = x / newpoints.length - masterShapes[i].name.length * 5
        y = (y / newpoints.length) - 12
        ctx.beginPath();
        ctx.font = "45px serif";
        ctx.textBaseline = 'top'
        ctx.fillStyle = '#000'
        let width = ctx.measureText(masterShapes[i].name).width + 10
        ctx.fillRect(x-5, y, width, parseInt("42px serif", 10));
        ctx.fillStyle = 'white'
        ctx.fillText(masterShapes[i].name, x, y);
        ctx.stroke();
    }
}

function getParentPoints () {
    var parentPoints = [];
    for (var i = 0; i < masterPoints.length; i++) {
        parentPoints.push(masterPoints[i]);
    }
    parentPoints.push(points);
    return parentPoints;
}

window.addEventListener('keyup', function(e) {
    if (e.key === 'Shift') {
        constrainAngles = false;
    }
});

document.querySelector('#clipboard').addEventListener('click', function(e) {
    e.preventDefault();
    clipboard("#clipboard");
});

canvas.addEventListener('dragover', function(e) {
    e.preventDefault();
});

canvas.addEventListener('wheel', function(e) {
    e.preventDefault()
    var delta = Math.sign(e.deltaY);
    zoom(delta);
});

// on canvas hover, if cursor is crosshair, draw line from last point to cursor
canvas.addEventListener('mousemove', function(e) {
    var x = getScaledCoords(e)[0];
    var y = getScaledCoords(e)[1];
    // round
    x = Math.round(x);
    y = Math.round(y);

    // update x y coords
    var xcoord = document.querySelector('#x');
    var ycoord = document.querySelector('#y');

    if(constrainAngles) {
        var lastPoint = points[points.length - 1];
        var dx = x - lastPoint[0];
        var dy = y - lastPoint[1];
        var angle = Math.atan2(dy, dx);
        var length = Math.sqrt(dx * dx + dy * dy);
        const snappedAngle = Math.round(angle / radiansPer45Degrees) * radiansPer45Degrees;
        var new_x = lastPoint[0] + length * Math.cos(snappedAngle);
        var new_y = lastPoint[1] + length * Math.sin(snappedAngle);
        x = Math.round(new_x);
        y = Math.round(new_y);
    }

    xcoord.innerHTML = x;
    ycoord.innerHTML = y;

    if (canvas.style.cursor == 'crosshair') {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);

        drawAllPolygons();

        for (var i = 0; i < points.length - 1; i++) {
            ctx.strokeStyle = rgb_color;
            drawLine(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]);
            ctx.stroke();
            // draw arc around each point
            ctx.beginPath();
            ctx.arc(points[i][0], points[i][1], 5, 0, 2 * Math.PI);
            // fill with white
            ctx.fillStyle = 'white';
            ctx.fill();
            ctx.stroke();
        }


        if ((points.length > 0 && drawMode == "polygon") || (points.length > 0 && points.length < 2 && drawMode == "line")) {
            ctx.strokeStyle = rgb_color;
            drawLine(points[points.length - 1][0], points[points.length - 1][1], x, y);
            ctx.stroke(); // new
            ctx.beginPath();
            ctx.arc(points[i][0], points[i][1], 5, 0, 2 * Math.PI);
            // fill with white
            ctx.fillStyle = 'white';
            ctx.fill();
            ctx.stroke();
        }
    }
});

// canvas.addEventListener('drop', function(e) {
//     e.preventDefault();
//     var file = e.dataTransfer.files[0];
//     var reader = new FileReader();
//
//     reader.onload = function(event) {
//         // only allow image files
//         img.src = event.target.result;
//     };
//     reader.readAsDataURL(file);
//
//     var mime_type = file.type;
//
//     if (
//         mime_type != 'image/png' &&
//         mime_type != 'image/jpeg' &&
//         mime_type != 'image/jpg'
//     ) {
//         alert('Only PNG, JPEG, and JPG files are allowed.');
//         return;
//     }
//
//     img.onload = function() {
//         scaleFactor = 0.5;
//         canvas.style.width = img.width * scaleFactor + 'px';
//         canvas.style.height = img.height * scaleFactor + 'px';
//         canvas.width = img.width;
//         canvas.height = img.height;
//         canvas.style.borderRadius = '10px';
//         ctx.drawImage(img, 0, 0);
//     };
//     // show coords
//     document.getElementById('coords').style.display = 'inline-block';
// });

function writePoints(parentPoints) {
    var normalized = [];

    // if normalized is true, normalize all points
    var imgHeight = img.height;
    var imgWidth = img.width;
    if (showNormalized) {
        for (var i = 0; i < parentPoints.length; i++) {
            var normalizedPoints = [];
            for (var j = 0; j < parentPoints[i].length; j++) {
                normalizedPoints.push([
                    Math.round(parentPoints[i][j][0] / imgWidth * 100) / 100,
                    Math.round(parentPoints[i][j][1] / imgHeight * 100) / 100
                ]);
            }
            normalized.push(normalizedPoints);
        }
        parentPoints = normalized;
    }

    // clean empty points
    parentPoints = parentPoints.filter(points => !!points.length);

    if (!parentPoints.length) {
        document.querySelector('#python').innerHTML = '';
        document.querySelector('#json').innerHTML;
        return;
    }

//     // create np.array list
//     var code_template = `
// [
// ${parentPoints.map(function(points) {
// return `([
// ${points.map(function(point) {
//     return `[${point[0]}, ${point[1]}]`;
// }).join(',')}
// ])`;
// }).join(',')}
// ]
//     `;
//
//     document.querySelector('#python').innerHTML = code_template;
//
//     var json_template = `
// {
// ${parentPoints.map(function(points) {
// return `[
// ${points.map(function(point) {
// return `{"x": ${point[0]}, "y": ${point[1]}}`;
// }).join(',')}
// ]`;
// }).join(',')}
// }
//     `;
//     document.querySelector('#json').innerHTML = json_template;
}

canvas.addEventListener('click', function(e) {
    // set cursor to crosshair
    canvas.style.cursor = 'crosshair';

    var x = getScaledCoords(e)[0];
    var y = getScaledCoords(e)[1];
    x = Math.round(x);
    y = Math.round(y);

    if(constrainAngles) {
        var lastPoint = points[points.length - 1];
        var dx = x - lastPoint[0];
        var dy = y - lastPoint[1];
        var angle = Math.atan2(dy, dx);
        var length = Math.sqrt(dx * dx + dy * dy);
        const snappedAngle = Math.round(angle / radiansPer45Degrees) * radiansPer45Degrees;
        var new_x = lastPoint[0] + length * Math.cos(snappedAngle);
        var new_y = lastPoint[1] + length * Math.sin(snappedAngle);
        x = Math.round(new_x);
        y = Math.round(new_y);
    }

    if (points.length > 2 && drawMode == "polygon") {
        let distX = x - points[0][0];
        let distY = y - points[0][1];
        // stroke is 3px and centered on the circle (i.e. 1/2 * 3px) and arc radius is
        if(Math.sqrt(distX * distX + distY * distY) <= 6.5) {
            closePath();
            return;
        }
    }

    points.push([x, y]);

    ctx.beginPath();
    ctx.strokeStyle = rgb_color;
    ctx.arc(x, y, 5, 0, 2 * Math.PI);
    // fill with white
    ctx.fillStyle = 'white';
    ctx.fill();
    ctx.stroke();

    if(drawMode == "line" && points.length == 2) {
        closePath();
    }
    else {
        ctx.beginPath();
        ctx.strokeStyle = rgb_color;
    }

    // ctx.arc(x, y, 155, 0, 2 * Math.PI);
    // concat all points into one array
    var parentPoints = [];

    for (var i = 0; i < masterPoints.length; i++) {
        parentPoints.push(masterPoints[i]);
    }
    // add "points"
    if(points.length > 0) {
        parentPoints.push(points);
    }

    writePoints(parentPoints);
});

function setDrawMode(mode) {
    drawMode = mode
    canvas.style.cursor = 'crosshair';
    document.querySelectorAll('.t-mode').forEach(el => el.classList.remove('active'))
    document.querySelector(`#mode-${mode}`).classList.add('active')
}

document.querySelector('#mode-polygon').addEventListener('click', function(e) {
    setDrawMode('polygon')
})

document.querySelector('#mode-line').addEventListener('click', function(e) {
    setDrawMode('line')
})

document.addEventListener('keydown', function(e) {
    if (e.key == 'l') {
        setDrawMode('line')
    }
    if (e.key == 'p') {
        setDrawMode('polygon')
    }
})

function draw () {

    drawAllPolygons()
    var parentPoints = getParentPoints()
    writePoints(parentPoints)
}

function highlightButtonInteraction (buttonId) {
    document.querySelector(buttonId).classList.add('active')
    setTimeout(() => document.querySelector(buttonId).classList.remove('active'), 100)
}

function undo () {
    highlightButtonInteraction('#undo')

    points.pop()
    draw()
}

document.querySelector('#undo').addEventListener('click', function(e) {
    undo()
})

function discardCurrentPolygon () {
    highlightButtonInteraction('#discard-current')

    points = []
    draw()
}

function discardPreviousPolygon () {
    masterShapes.pop()
    masterColors.pop()
    masterPoints.pop()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0)
    drawAllPolygons()
    updatePythonOutput()
}


document.querySelector('#discard-current').addEventListener('click', function(e) {
    discardCurrentPolygon()
})

function clearAll() {
    highlightButtonInteraction('#clear')

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0)

    points = []
    masterPoints = []
    masterShapes = []
    masterColors = []
    updatePythonOutput()
}

document.querySelector('#clear').addEventListener('click', function(e) {
    clearAll()
})

function saveImage () {
    highlightButtonInteraction('#save-image')

    var link = document.createElement('a')
    link.download = 'image.png'
    link.href = canvas.toDataURL()
    link.click()
}

document.querySelector('#save-image').addEventListener('click', function(e) {
    saveImage()
})

function completeCurrentPolygon () {

    let name = prompt("Введите имя объекта:")
    if (name && points.length > 0) {
        masterShapes.push({
            name: name, // Сохраняем имя объекта
            points: points.slice() // последняя добавленная коллекция точек
        })
    }

    canvas.style.cursor = 'default'

    // save current polygon points
    masterPoints.push(points)
    points = []

    // dont choose a color that has already been chosen
    var remaining_choices = color_choices.filter(function(x) {
        return !masterColors.includes(x)
    });

    if (remaining_choices.length == 0) {
        remaining_choices = color_choices
    }

    rgb_color = remaining_choices[Math.floor(Math.random() * remaining_choices.length)]
    masterColors.push(rgb_color)

    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0)
    drawAllPolygons()

    updatePythonOutput()

    draw()
}

window.addEventListener('keydown', function(e) {
    if (e.key === 'c' && (e.ctrlKey || e.metaKey)) {
        discardPreviousPolygon ()
    }

    if (e.key === 'z' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        e.stopImmediatePropagation()

        undo()
    }

    if (e.key === 'Shift') {
        constrainAngles = true;
    }

    if (e.key === 'Escape') {
        discardCurrentPolygon()
    }

    if (e.key === 'e' && (e.ctrlKey || e.metaKey)) {
        clearAll()
    }

    if (e.key === 's' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault()
        e.stopImmediatePropagation()

        saveImage()
    }

    if (e.key === 'Enter') {
        // Проверяем, нужно ли завершать текущее рисование (например, режим рисования полигона)
            completeCurrentPolygon(); // Вызываем функцию завершения полигона, если есть достаточно точек
    }
})