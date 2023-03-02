def simulate_ripples(img):
    """
    simulate ripples (periodic noises in an image)
    img (np.array): greyscale img with no noises e.g. np.ones((50,50))*125
    """
    rows,columns = img.shape
    rowVector = np.arange(rows).T
    period = 10
    amplitude = 0.5 #magnitude of the ripples
    offset = 1-amplitude #how much the cosine is raised above 0 to show the 'peaks'
    cosVector = amplitude*(1+np.cos(2*np.pi*rowVector/period))/2 + offset
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(rowVector.T,cosVector.T)
    ripplesImage = np.repeat(cosVector[:,np.newaxis],columns,axis=1) #repeat across columns
    #multiply the ripples by the image to get an image with periodic noise in it
    grayImage = ripplesImage*img
    im = ax[1].imshow(grayImage,cmap='gray')
    plt.colorbar(im,ax=ax[1])
    return grayImage

noiseImage = simulate_ripples(np.ones((50,50))*125)

def remove_periodic_noise(img,amplitudeThreshold=20):
    """
    img (np.array): greyscale img
    amplitudeThreshold (float): to threshold the amplitude image
    -------
    pipeline:
    1. convolve the image with a box filter to remove noise
    2. perform 2D FFT to get the frequency image and shift it such that the DC is in the center
    3. Obtain the amplitude image
    4. Threshold the amplitude image to identify bright peaks which represents periodic noises
    5. Remove central DC spike
    6. Remove bright spikes from the frequency image
    7. Perform inverse FFT on the frequency image to restore the image without the periodic noises
    Note:
    In real life, if image is noises, it's hard to identify the periodic noises (aka bright spots)
    """
    nrow,ncol = img.shape[0],img.shape[1]
    w = 10
    h = 5
    kernel = np.ones((h,w))/(h*w)
    k = np.pad(kernel,((3,3),(3,3)))
    # convolve kernel (box filter) with the image to remove noise
    filteredImage = scipy.signal.convolve2d(img,k,mode='same')
    # filteredImage = img
    frequencyImage = np.fft.fftshift(np.fft.fft2(filteredImage))
    fig, axes = plt.subplots(2,3,figsize=(10,5))
    axes[0,0].imshow(frequencyImage.astype(np.uint8))
    axes[0,0].set_title('Frequency Image')
    amplitudeImage = np.abs(frequencyImage)
    axes[0,1].imshow(amplitudeImage.astype(np.uint8))
    axes[0,1].set_title('Amplitude Image')
    brightSpikes = amplitudeImage > amplitudeThreshold #boolean image
    
    axes[0,2].imshow(filteredImage)
    axes[0,2].set_title('Original Img w box filter')
    #exclude the central DC spike
    brightSpikes[nrow//2-2:nrow//2+2,ncol//2-2:ncol//2+2] = False
    
    axes[1,0].imshow(brightSpikes)
    axes[1,0].set_title(f'Bright spikes with threshold: {amplitudeThreshold}')

    #remove the periodic noises by setting it to 0
    frequencyImage[brightSpikes] = 0
    
    axes[1,1].imshow(frequencyImage.astype(np.uint8))
    axes[1,1].set_title('Removed bright spots')

    #perform inverse FFT to get back image
    filteredImage = np.fft.ifft2(np.fft.fftshift(frequencyImage))
    amplitudeImage = np.abs(filteredImage)
    
    axes[1,2].imshow(amplitudeImage.astype(np.uint8))
    axes[1,2].set_title('Inverse FFT after removing bright spots')
    plt.tight_layout()
    return