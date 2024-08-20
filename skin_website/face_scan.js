document.addEventListener('DOMContentLoaded', function() {
    const video = document.getElementById('video');
    const captureButton = document.getElementById('captureButton');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    const resultDiv = document.getElementById('result');

    navigator.mediaDevices.getUserMedia({ video: true })
        .then(stream => {
            video.srcObject = stream;
            video.play();
        })
        .catch(err => {
            console.error('Error accessing webcam: ', err);
        });

    captureButton.addEventListener('click', function() {
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL('image/png');
        resultDiv.innerHTML = `<img src="${imageData}" alt="Captured Image">`;
    });
});
