// import React, { useEffect, useState, useRef } from 'react';
// import axios from 'axios';

// function App() {
//     const [images, setImages] = useState([]);
//     const [currentIndex, setCurrentIndex] = useState(0);
//     const imageContainerRef = useRef(null);

//     useEffect(() => {
//         // Gọi API để lấy danh sách ảnh
//         axios.get('http://localhost:5000/api/images')
//             .then(response => {
//               setImages(response.data);
//             })
//             .catch(error => {
//                 console.error('Error fetching images:', error);
//             });
//     }, []);

//     // useEffect(() => {
//     //     const handleScroll = (event) => {
//     //         if (event.deltaY > 0) {
//     //             setCurrentIndex(prevIndex => Math.min(prevIndex + 1, images.length - 1));
//     //         } else if (event.deltaY < 0) {
//     //             setCurrentIndex(prevIndex => Math.max(prevIndex - 1, 0));
//     //         }
//     //     };

//     //     window.addEventListener('wheel', handleScroll);

//     //     // Cleanup
//     //     return () => {
//     //         window.removeEventListener('wheel', handleScroll);
//     //     };
//     // }, [images]);

//     useEffect(() => {
//         const handleScroll = (event) => {
//             if (event.deltaY > 0) {
//                 setCurrentIndex(prevIndex => Math.min(prevIndex + 1, images.length - 1));
//             } else if (event.deltaY < 0) {
//                 setCurrentIndex(prevIndex => Math.max(prevIndex - 1, 0));
//             }
//         };

//         const imageContainer = imageContainerRef.current;
//         if (imageContainer) {
//             imageContainer.addEventListener('wheel', handleScroll);
//         }

//         // Cleanup
//         return () => {
//             if (imageContainer) {
//                 imageContainer.removeEventListener('wheel', handleScroll);
//             }
//         };
//     }, [images]);

//     return (
//         // <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
//         //     {images.length > 0 && (
//         //         <img src={images[currentIndex]} alt={`Photo ${currentIndex + 1}`} style={{ maxWidth: '80%', maxHeight: '80%' }} />
//         //     )}
//         // </div>
//         <div 
//             ref={imageContainerRef} 
//             style={{ 
//                 display: 'flex', 
//                 justifyContent: 'center', 
//                 alignItems: 'center', 
//                 height: '100vh', 
//                 overflow: 'hidden' 
//             }}
//         >
//             {images.length > 0 && (
//                 <img 
//                     src={images[currentIndex]} 
//                     alt={`Photo ${currentIndex + 1}`} 
//                     style={{ maxWidth: '80%', maxHeight: '80%' }} 
//                 />
//             )}
//         </div>
//     );
// }

// export default App;



// App.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
    const [images, setImages] = useState([]); // Danh sách ảnh từ API
    const [currentIndex, setCurrentIndex] = useState(0); // Ảnh hiện tại

    useEffect(() => {
        // Gọi API để lấy danh sách URL ảnh
        axios.get('http://localhost:8000/convert/1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd')
            .then(response => {
                setImages(response.data.images); // Thiết lập danh sách ảnh
            })
            .catch(error => {
                console.error('Error fetching images:', error);
            });
    }, []);

    // Xử lý cuộn chuột để chuyển ảnh
    const handleScroll = (event) => {
        if (event.deltaY > 0) {
            // Cuộn xuống: Hiển thị ảnh kế tiếp
            setCurrentIndex(prevIndex => Math.min(prevIndex + 1, images.length - 1));
        } else if (event.deltaY < 0) {
            // Cuộn lên: Hiển thị ảnh trước đó
            setCurrentIndex(prevIndex => Math.max(prevIndex - 1, 0));
        }
    };

    return (
        <div
            style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh', overflow: 'hidden' }}
            onWheel={handleScroll} // Sự kiện cuộn chuột trong vùng chứa
        >
            {images.length > 0 ? (
                <img
                    src={images[currentIndex]}
                    alt={`Slice ${currentIndex + 1}`}
                    style={{ maxWidth: '80%', maxHeight: '80%' }}
                />
            ) : (
                <p>Loading images...</p>
            )}
        </div>
    );
}

export default App;
