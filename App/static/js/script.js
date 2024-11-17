document.addEventListener("DOMContentLoaded", function() {
    const fileInput = document.querySelector("input[type='file']");
    fileInput.addEventListener("change", function() {
        const videoElement = document.getElementById("videos");
        const source = document.getElementById("video_source");
        const file = this.files[0];
        if (file) {
            const fileURL = URL.createObjectURL(file);
            source.src = fileURL;
            videoElement.load();
        }
    });
});
