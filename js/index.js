// import * as humanseg from '../src/index';

import * as humanseg from '@paddlejs-models/humanseg';

async function load() {
    await humanseg.load();
    document.getElementById('loading').style.display = 'none';
}

load();
const canvas1 = document.getElementById('demo1');
const canvas2 = document.getElementById('demo2');

const ctx = canvas1.getContext('2d');
const img = new Image();
img.src = './bgImgs/bg.jpg';
img.onload = function () {
    ctx.drawImage(img, 0, 0, canvas1.width, canvas1.height);
};

async function run(input) {
    const {
        data
    } = await humanseg.getGrayValue(input);

    humanseg.drawHumanSeg(canvas1, data);
    humanseg.drawMask(canvas2, data, true);
}

function selectImage(file) {
    if (!file.files || !file.files[0]) {
        return;
    }
    const reader = new FileReader();
    reader.onload = function (evt) {
        const img = document.getElementById('image');
        img.src = evt.target.result;
        img.onload = function () {
            run(img);
        };
    };
    reader.readAsDataURL(file.files[0]);
}

// selectImage
document.getElementById('uploadImg').onchange = function () {
    selectImage(this);
};