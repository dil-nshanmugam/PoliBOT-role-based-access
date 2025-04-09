axios.post('https://localhost:8000/chat', {
    "question": userInput.toString()
}, {
    // Optional configuration
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer your_token_here'
    }
  })
  .then(response => {
    // Handle successful response
    console.log('Response data:', response);
    return response.answer;
  })
  .catch(error => {
    // Handle error
    console.error('Error occurred:', error);
  });