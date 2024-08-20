document.getElementById('signupForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent the default form submission

    // Retrieve form values
    const fullName = document.getElementById('fullName').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    // Validate password and confirm password fields
    if (password !== confirmPassword) {
        alert('Passwords do not match.');
        return;
    }

    // Simulate account creation process
    createAccount(fullName, email, password)
        .then(() => {
            window.location.href = 'login.html'; // Redirect to login page
        })
        .catch((error) => {
            alert('Failed to create account: ' + error.message); // Handle errors
        });
});

// Simulate account creation with a promise
function createAccount(fullName, email, password) {
    return new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve(); // Simulate successful account creation
            // Uncomment the following line to simulate a failed account creation
            // reject(new Error('Email already exists'));
        }, 1000);
    });
}
