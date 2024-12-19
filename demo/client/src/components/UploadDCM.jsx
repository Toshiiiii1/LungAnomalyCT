import { useState, useEffect } from "react";
import axios from "axios";

const UploadDCM = () => {
	const [files, setFiles] = useState([]);
	const [result, setResult] = useState([]); // Danh sách ảnh từ API
	const [currentIndex, setCurrentIndex] = useState(0); // Ảnh hiện tại
	const [predictIndex, setPredictIndex] = useState(0); // Ảnh hiện tại
	const [predictedImageUrl, setPredictedImageUrl] = useState(""); // URL ảnh đã xử lý
	const [predictedImageUrls, setPredictedImageUrls] = useState([]);
	const [isLoading, setIsLoading] = useState(false);

	const handleFolderUpload = (event) => {
		setFiles(Array.from(event.target.files));

		const folderData = files.map((file) => ({
			name: file.name,
			path: file.webkitRelativePath,
			size: file.size,
			type: file.type,
		}));
		console.log(files);
		console.log("Files in the folder:", folderData);
		// You can upload files to the server here
	};

	const handleUpload = async () => {
		if (!files || files.length === 0) {
			alert("Vui lòng chọn một thư mục!");
			return;
		}

		const formData = new FormData();
		files.forEach((file) => {
			formData.append("files", file);
		});

		setIsLoading(true);
		try {
			const response = await axios.post(
				"http://localhost:8000/upload_dcm",
				formData,
				{
					headers: { "Content-Type": "multipart/form-data" },
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
			setIsLoading(false);
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
		window.scrollTo({ top: 0});
	};

	return (
		<>
			<div className="app-container">
				{/* Main Content */}
				<div className="main-content">
					<div className="file_uploader">
						<h1>Tải lên một folder</h1>
						<input
							type="file"
							webkitdirectory="true"
							onChange={handleFolderUpload}
						/>
						<button onClick={handleUpload} disabled={isLoading}>
							{isLoading ? "Đang tải lên..." : "Tải lên"}
						</button>
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
										Lát cắt số {currentIndex + 1} trong số {result.length} lát cắt
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
											Ảnh dự đoán:{" "}
											{`Lát cắt ${predictIndex + 1}`}
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
								<h2>Các ảnh dự đoán</h2>
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
												? parseInt(
														sliceNumberMatch[1],
														10
												  )
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
																url,
																sliceNumber
															)
														}
													/>
													<p>
														Lát cắt số {sliceNumber + 1}
													</p>
												</div>
											);
										})
									) : (
										<p>Không có ảnh để hiển thị</p>
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

export default UploadDCM;
