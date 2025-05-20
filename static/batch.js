// batch.js - Batch processing functionality for Chassis/VIN Detector application

// These functions are called from the main script in index.html
// and need access to the variables defined there

// Process multiple files in batch
function processBatch() {
    const fileInput = document.getElementById('file-input');
    const loader = document.getElementById('loader');
    const previewImage = document.getElementById('preview-image');
    const resultsContainer = document.getElementById('results-container');
    const downloadContainer = document.getElementById('download-container');
    const resultsList = document.getElementById('results-list');
    const downloadLink = document.getElementById('download-link');

    if (fileInput.files.length <= 1) {
        return; // Use the regular process method for single files
    }
    
    // Show loader and hide previous results
    loader.style.display = 'block';
    previewImage.classList.add('d-none');
    resultsContainer.style.display = 'none';
    downloadContainer.style.display = 'none';
    
    // Create FormData with all files
    const formData = new FormData();
      // Add all files to formData
    for (let i = 0; i < fileInput.files.length; i++) {
        formData.append('file', fileInput.files[i]);
    }
    
    // Add form parameters
    const obscurePart = document.getElementById('obscure-part').value;
    const obscureMethod = document.getElementById('obscure-method').value;
    const blurAmount = document.getElementById('blur-amount').value;
    const mosaicSize = document.getElementById('mosaic-size').value;
    const debugMode = document.getElementById('debug-mode').checked;
    
    formData.append('obscure_part', obscurePart);
    formData.append('obscure_method', obscureMethod);
    formData.append('blur_amount', blurAmount);
    formData.append('mosaic_block_size', mosaicSize);
    if (debugMode) {
        formData.append('debug_mode', 'true');
    }    // Create an AbortController with a timeout
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000); // 3-minute timeout for batch processing
    
    fetch('/process_batch', {
        method: 'POST',
        body: formData,
        signal: controller.signal
    })
    .then(response => {
        clearTimeout(timeoutId);
        if (!response.ok) {
            throw new Error(`Server responded with status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => handleBatchResponse(data))
    .catch(error => {
        loader.style.display = 'none';
        alert('Error processing images: ' + error);
    });
}

// Handle the batch response from the server
function handleBatchResponse(data) {
    const loader = document.getElementById('loader');
    const previewImage = document.getElementById('preview-image');
    const resultsContainer = document.getElementById('results-container');
    const downloadContainer = document.getElementById('download-container');
    const resultsList = document.getElementById('results-list');
    const downloadLink = document.getElementById('download-link');
    
    loader.style.display = 'none';
    
    if (data.error) {
        alert('Error: ' + data.error);
        return;
    }
    
    // Display the batch results
    resultsList.innerHTML = '';
    
    // Add overall summary
    const summaryLi = document.createElement('li');
    summaryLi.className = 'list-group-item list-group-item-success';
    summaryLi.textContent = data.message;
    resultsList.appendChild(summaryLi);
    
    // Create a div to display thumbnails of processed images
    const thumbnailsDiv = document.createElement('div');
    thumbnailsDiv.className = 'd-flex flex-wrap gap-2 mt-3 mb-3';
    resultsList.appendChild(thumbnailsDiv);
    
    // Show a preview gallery of all processed images
    data.batch_results.forEach((result, index) => {
        if (result.status === 'success') {
            // Create a thumbnail container for each image
            const thumbContainer = document.createElement('div');
            thumbContainer.className = 'card text-center';
            thumbContainer.style.width = '150px';
            
            // Create the image thumbnail
            const img = document.createElement('img');
            img.src = result.image_preview;
            img.alt = `Processed ${result.filename}`;
            img.className = 'card-img-top';
            img.style.maxHeight = '100px';
            img.style.objectFit = 'contain';
            
            // Create a download link
            const downloadBtn = document.createElement('a');
            downloadBtn.href = result.image_preview;
            downloadBtn.download = `processed_${result.filename}`;
            downloadBtn.className = 'btn btn-sm btn-primary mt-1 mb-1';
            downloadBtn.innerHTML = '<i class="bi bi-download"></i>';
            
            // Create a title
            const title = document.createElement('div');
            title.className = 'card-footer text-truncate small';
            title.title = result.filename;
            title.textContent = result.filename;
            
            // Add elements to the container
            thumbContainer.appendChild(img);
            thumbContainer.appendChild(downloadBtn);
            thumbContainer.appendChild(title);
            
            // Add to the thumbnails div
            thumbnailsDiv.appendChild(thumbContainer);
            
            // If it's the first image, also show it in the main preview
            if (index === 0) {
                previewImage.src = result.image_preview;
                previewImage.classList.remove('d-none');
                
                downloadLink.href = result.image_preview;
                downloadLink.download = `processed_${result.filename}`;
                downloadContainer.style.display = 'block';
            }
        } else {
            // Add error message for failed images
            const errorLi = document.createElement('li');
            errorLi.className = 'list-group-item list-group-item-danger';
            errorLi.textContent = `Failed to process ${result.filename}: ${result.message}`;
            resultsList.appendChild(errorLi);
        }
    });
    
    // Display results container
    resultsContainer.style.display = 'block';
}
