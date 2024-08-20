document.addEventListener('DOMContentLoaded', () => {
    // Elements for Skin Analysis
    const imageUpload = document.getElementById('imageUpload');
    const analyzeButton = document.getElementById('analyzeButton');
    const resultsDiv = document.getElementById('results');
    const recommendationsDiv = document.getElementById('recommendations');

    analyzeButton.addEventListener('click', () => {
        const file = imageUpload.files[0];
        // Here you would typically send the image data to a server for analysis
        // For this example, we'll simulate results
        const simulatedResults = {
            skinType: 'Oily',
            concerns: ['acne', 'blackheads']
        };

        resultsDiv.innerHTML = `
            <h2>Skin Analysis Results</h2>
            <p>Skin Type: ${simulatedResults.skinType}</p>
            <p>Concerns: ${simulatedResults.concerns.join(', ')}</p>
        `;

        // Simulate product recommendations based on results
        const recommendedProducts = ['Oil-free moisturizer', 'Acne treatment'];
        recommendationsDiv.innerHTML = `
            <h2>Recommended Products</h2>
            <ul>
                ${recommendedProducts.map(product => `<li>${product}</li>`).join('')}
            </ul>
        `;
    });

    // Slider functionality
    const slides = document.querySelector('.slides');
    const slideCount = document.querySelectorAll('.slide').length;
    let currentIndex = 0;
    const slideInterval = 3000; // Change slide every 3 seconds

    function moveToNextSlide() {
        currentIndex = (currentIndex + 1) % slideCount;
        const offset = -currentIndex * 100;
        slides.style.transform = `translateX(${offset}%)`;
    }

    // Change slide automatically every 3 seconds
    setInterval(moveToNextSlide, slideInterval);

    // Popup functionality
    const loginButton = document.querySelector('.nav-right .button[href="#login"]');
    const signupButton = document.querySelector('.nav-right .button[href="#signup"]');
    const popup = document.createElement('div');
    popup.className = 'popup';
    popup.innerHTML = `
        <div class="popup-content">
            <span class="close">&times;</span>
            <iframe id="popup-frame" src="" frameborder="0"></iframe>
        </div>
    `;
    document.body.appendChild(popup);

    const popupFrame = document.getElementById('popup-frame');
    const closeBtn = document.querySelector('.popup .close');

    loginButton.addEventListener('click', (e) => {
        e.preventDefault();
        popupFrame.src = 'login.html'; // Load login page in the popup
        popup.style.display = 'block';
    });

    signupButton.addEventListener('click', (e) => {
        e.preventDefault();
        popupFrame.src = 'signup.html'; // Load signup page in the popup
        popup.style.display = 'block';
    });

    closeBtn.addEventListener('click', () => {
        popup.style.display = 'none';
    });

    // Close popup if clicked outside of it
    window.addEventListener('click', (e) => {
        if (e.target === popup) {
            popup.style.display = 'none';
        }
    });
});

document.addEventListener('DOMContentLoaded', () => {
    // Elements for Skin Analysis
    const imageUpload = document.getElementById('imageUpload');
    const analyzeButton = document.getElementById('analyzeButton');
    const resultsDiv = document.getElementById('results');
    const recommendationsDiv = document.getElementById('recommendations');

    analyzeButton.addEventListener('click', () => {
        const file = imageUpload.files[0];
        // Here you would typically send the image data to a server for analysis
        // For this example, we'll simulate results
        const simulatedResults = {
            skinType: 'Oily',
            concerns: ['acne', 'blackheads']
        };

        resultsDiv.innerHTML = `
            <h2>Skin Analysis Results</h2>
            <p>Skin Type: ${simulatedResults.skinType}</p>
            <p>Concerns: ${simulatedResults.concerns.join(', ')}</p>
        `;

        // Simulate product recommendations based on results
        const recommendedProducts = ['Oil-free moisturizer', 'Acne treatment'];
        recommendationsDiv.innerHTML = `
            <h2>Recommended Products</h2>
            <ul>
                ${recommendedProducts.map(product => `<li>${product}</li>`).join('')}
            </ul>
        `;
    });

    // Slider functionality
    const slides = document.querySelector('.slides');
    const slideCount = document.querySelectorAll('.slide').length;
    let currentIndex = 0;
    const slideInterval = 3000; // Change slide every 3 seconds

    function moveToNextSlide() {
        currentIndex = (currentIndex + 1) % slideCount;
        const offset = -currentIndex * 100;
        slides.style.transform = `translateX(${offset}%)`;
    }

    // Change slide automatically every 3 seconds
    setInterval(moveToNextSlide, slideInterval);
});

