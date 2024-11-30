import { useState, useEffect } from "react";
import axios from "axios";

const Blogs2 = () => {
	const [file, setFile] = useState(null);
	const [previewUrl, setPreviewUrl] = useState("");
	const [originalImageUrl, setOriginalImageUrl] = useState("");
	const [predictedImageUrl, setPredictedImageUrl] = useState("");
	const [isLoading, setIsLoading] = useState(false);

	const handleFileChange = (e) => setFile(e.target.files[0]);

	const handleUpload = async () => {
		if (!file) {
			alert("Vui lòng chọn một tệp ảnh!");
			return;
		}
		setPreviewUrl(URL.createObjectURL(file));
		
		const formData = new FormData();
		formData.append("file", file);

		setIsLoading(true);
		try {
			const response = await axios.post(
				"http://localhost:8000/upload_png_jpg_jpeg",
				formData,
				{
					headers: { "Content-Type": "multipart/form-data" },
				}
			);
			setOriginalImageUrl(response.data.original_image); // URL ảnh gốc từ server
			setPredictedImageUrl(response.data.predicted_image); // URL ảnh dự đoán từ server
		} catch (error) {
			console.error("Upload failed:", error);
			alert("Tải lên thất bại.");
		} finally {
			setIsLoading(false);
		}
	};

	return (
		<>
			<div className="app-container">
				{/* Main Content */}
				<div className="main-content">
					<div className="file_uploader">
						<h1>Tải lên một file ảnh</h1>
						<input
							type="file"
							accept=".png,.jpg,.jpeg"
							onChange={handleFileChange}
						/>
						<button onClick={handleUpload} disabled={isLoading}>
							{isLoading ? "Đang tải lên..." : "Tải lên"}
						</button>
					</div>
					{/* Hiển thị kết quả */}
					<div
						style={{
							display: "flex",
							justifyContent: "center",
							alignItems: "center",
						}}
					>
						{/* Ảnh hiển thị */}
						{previewUrl && (
							<div>
								<img
									src={previewUrl}
									alt={"Original Image"}
									style={{
										margin: "20px", // Thêm khoảng cách bên trái và bên phải
									}}
								/>
								<p
									style={{
										marginTop: "20px",
										fontSize: "18px",
									}}
								>
									Original Image
								</p>
							</div>
						)}
						{predictedImageUrl && (
							<div style={{ textAlign: "center" }}>
								<img
									src={predictedImageUrl}
									alt="Predicted"
									style={{
										margin: "20px", // Thêm khoảng cách bên trái và bên phải
									}}
								/>
								<p
									style={{
										marginTop: "20px",
										fontSize: "18px",
									}}
								>
									Predicted Image
								</p>
							</div>
						)}
					</div>
				</div>
			</div>
		</>
	);
};

export default Blogs2;
