document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    // Retrieve form values
    const email = document.getElementById('loginEmail').value;
    const password = document.getElementById('loginPassword').value;

    // Simulate server request for user login
    loginUser(email, password)
        .then(() => {
            window.location.href = 'face-scanning.html'; // Redirect to face scanning page
        })
        .catch((error) => {
            alert('Failed to log in: ' + error.message); // Handle errors
        });
});

// Simulate user login with a promise
function loginUser(email, password) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve(); // Simulate successful login
            // Uncomment the following line to simulate a failed login
            // reject(new Error('Invalid email or password'));
        }, 1000);
    });
}
