import { useState, useEffect } from "react";
import axios from "axios";

const Blogs = () => {
	const [file1, setFile1] = useState(null);
	const [file2, setFile2] = useState(null);
	const [result, setResult] = useState([]); // Danh sách ảnh từ API
	const [currentIndex, setCurrentIndex] = useState(0); // Ảnh hiện tại
	const [predictIndex, setPredictIndex] = useState(0); // Ảnh hiện tại
	const [predictedImageUrl, setPredictedImageUrl] = useState(""); // URL ảnh đã xử lý
	const [predictedImageUrls, setPredictedImageUrls] = useState([]);
	const [isLoading, setIsLoading] = useState(false);

	const handleFile1Change = (e) => setFile1(e.target.files[0]);
	const handleFile2Change = (e) => setFile2(e.target.files[0]);

	const handleUpload = async () => {
		if (!file1 || !file2) {
			alert("Vui lòng chọn cả hai tệp!");
			return;
		}

		const formData = new FormData();
		formData.append("file1", file1);
		formData.append("file2", file2);

		setIsLoading(true);
		try {
			const response = await axios.post(
				"http://localhost:8000/upload_mhd_raw",
				formData,
				{
					headers: {
						"Content-Type": "multipart/form-data",
					},
				}
			);
			setResult(response.data.files);
			setPredictedImageUrls(response.data.pred_files);
			setCurrentIndex(0);
			setPredictedImageUrl("");
		} catch (error) {
			console.error("Upload failed:", error);
			alert("Tải lên thất bại.");
		} finally {
			setIsLoading(false); // Kết thúc tải ảnh
		}
	};

	// Xử lý cuộn chuột để chuyển ảnh
	const handleScroll = (event) => {
		if (event.deltaY > 0) {
			// Cuộn xuống: Hiển thị ảnh kế tiếp
			setCurrentIndex((prevIndex) =>
				Math.min(prevIndex + 1, result.length - 1)
			);
		} else if (event.deltaY < 0) {
			// Cuộn lên: Hiển thị ảnh trước đó
			setCurrentIndex((prevIndex) => Math.max(prevIndex - 1, 0));
		}
	};

	const handleImageClick = (url, sliceNumber) => {
		setPredictedImageUrl(url);
		setCurrentIndex(sliceNumber);
		setPredictIndex(sliceNumber);
	};

	return (
		<>
			<div className="app-container">
				{/* Main Content */}
				<div className="main-content">
					<div className="file_uploader">
						<h1>Tải lên 2 file .mhd và .raw</h1>
						<input
							type="file"
							accept=".mhd"
							onChange={handleFile1Change}
						/>
						<input
							type="file"
							accept=".raw"
							onChange={handleFile2Change}
						/>
						<button onClick={handleUpload}>Tải lên</button>
					</div>
					{isLoading ? (
						<p>Đang tải ảnh...</p>
					) : result.length > 0 ? (
						<>
							{/* Khung chứa ảnh hiển thị và ảnh dự đoán */}
							<div
								style={{
									display: "flex",
									justifyContent: "center",
									alignItems: "center",
								}}
							>
								{/* Ảnh hiển thị */}
								<div>
									<img
										src={result[currentIndex]}
										alt={`Slice ${currentIndex}`}
										style={{
											margin: "20px", // Thêm khoảng cách bên trái và bên phải
										}}
										onWheel={handleScroll}
									/>
									<p
										style={{
											marginTop: "20px",
											fontSize: "18px",
										}}
									>
										Slice {currentIndex} of {result.length}
									</p>
								</div>

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
											Predicted Image:{" "}
											{`Slice ${predictIndex}`}
										</p>
									</div>
								)}
							</div>

							{/* Các nút điều khiển Previous và Next */}
							<div style={{ marginTop: "20px" }}>
								<button
									onClick={() =>
										setCurrentIndex((prevIndex) =>
											prevIndex > 0
												? prevIndex - 1
												: result.length - 1
										)
									}
									style={{
										marginRight: "10px",
										padding: "10px",
										fontSize: "16px",
										cursor: "pointer",
									}}
								>
									Previous
								</button>
								<button
									onClick={() =>
										setCurrentIndex((prevIndex) =>
											prevIndex < result.length - 1
												? prevIndex + 1
												: 0
										)
									}
									style={{
										marginLeft: "10px",
										padding: "10px",
										fontSize: "16px",
										cursor: "pointer",
									}}
								>
									Next
								</button>
								{/* <button
									onClick={handlePredict}
									style={{
										marginLeft: "10px",
										padding: "10px",
										fontSize: "16px",
										cursor: "pointer",
									}}
								>
									Predict
								</button> */}
							</div>
							<div>
								<h2>Predicted Images</h2>
								<div
									style={{
										display: "flex",
										flexWrap: "wrap",
										gap: "10px",
										justifyContent: "center",
									}}
								>
									{predictedImageUrls.length > 0 ? (
										predictedImageUrls.map((url, index) => {
											// Trích xuất số thứ tự mặt cắt từ URL
											const sliceNumberMatch =
												url.match(/_slice_(\d+)\.png$/);
											const sliceNumber = sliceNumberMatch
												? parseInt(sliceNumberMatch[1], 10)
												: "Unknown";

											return (
												<div
													key={index}
													style={{
														textAlign: "center",
													}}
												>
													<img
														src={url}
														alt={`Slice ${sliceNumber}`}
														style={{
															maxWidth: "300px",
															maxHeight: "300px",
															cursor: "pointer",
														}}
														onClick={() =>
															handleImageClick(
																url, sliceNumber
															)
														}
													/>
													<p>
														Image {index + 1} -
														Slice {sliceNumber}
													</p>
												</div>
											);
										})
									) : (
										<p>No images to display</p>
									)}
								</div>
							</div>
						</>
					) : (
						<p>Không có ảnh để hiển thị</p>
					)}
				</div>
			</div>
		</>
	);
};

export default Blogs;
