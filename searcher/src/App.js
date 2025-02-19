import React, { useState } from 'react';
import './App.css';

function App() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageUrl, setImageUrl] = useState('');
  const [searchResults, setSearchResults] = useState([]);

  const handleSearchChange = (e) => {
    setSearchQuery(e.target.value);
  };

  const handleImageChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const imageFile = e.target.files[0];
      setSelectedImage(imageFile);
      const url = URL.createObjectURL(imageFile);
      setImageUrl(url);
      setSearchQuery('');
    }
  };

  // React 컴포넌트 내에서
  const handleSubmit = (e) => {
    e.preventDefault();

    const formData = new FormData();
    formData.append('file', selectedImage);
    formData.append('searchText', searchQuery);

    fetch('http://127.0.0.1:8024/upload', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        setSearchResults(data.results);
        console.log(searchResults);
    })
    .catch(error => {
        console.error('Error:', error);
    });
  };


  return (
    <div className="App">
      <form onSubmit={handleSubmit}>
        {imageUrl && (
          <img src={imageUrl} alt="Uploaded" />
        )}
        <input
          type="text"
          placeholder="Input your question..."
          value={searchQuery}
          onChange={handleSearchChange}
        />
        <button type="submit">Search</button>
        <input
          id="image-upload"
          type="file"
          onChange={handleImageChange}
          accept="image/*"
          style={{ display: 'none' }}
        />
        <label htmlFor="image-upload" className="custom-file-upload">
          Upload image
        </label>
      </form>
      {searchResults.length > 0 && (<table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Score</th>
            <th>Document</th>
          </tr>
        </thead>
        <tbody>
          {searchResults.map((result, index) => (
            <tr key={index}>
              <td>{result.rank}</td>
              <td>{result.score}</td>
              <td>{result.text}</td>
            </tr>
          ))}
        </tbody>
      </table>)}
    </div>
  );
}

export default App;
