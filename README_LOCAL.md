# Chạy VoxCPM trên máy này

Tool tạo và lưu giọng **Giọng Nhà**: nhấp đúp `start-voice-studio.cmd`, rồi mở
**http://localhost:8809**. Có ba giọng mẫu tiếng Việt, thư viện giọng lưu lâu dài
và chức năng đọc văn bản bằng giọng đã chọn. Xem `voice_studio/README.md`.
Nhấp đúp `stop-voice-studio.cmd` để tắt tool và mô hình.

Mở **http://127.0.0.1:8808** để dùng giao diện.

- Khởi chạy lại: nhấp đúp `start-voxcpm.cmd`.
- Tắt ứng dụng: nhấp đúp `stop-voxcpm.cmd`.
- Nhật ký: `.runtime/server.log` và `.runtime/server-error.log`.
- Âm thanh tiếng Việt đã tạo thử: `output/test-vi.wav`.

Môi trường riêng dùng Python 3.11.17, PyTorch 2.10.0 + CUDA 12.8,
TorchAudio 2.10.0 + CUDA 12.8, TorchCodec 0.10.0 và FFmpeg shared 8.1.
Python hệ thống được giữ nguyên. Các thư viện của repo được cài trong `.venv`.
Python 3.11 nằm ở `../.python`; cần giữ thư mục này nếu tiếp tục dùng môi trường hiện tại.

Trọng số VoxCPM2 nằm trong `pretrained_models/VoxCPM2`, lấy từ bản Hugging Face
`32279effe8c19989596f05d353d1447f51d9e915`. Tệp `model.safetensors` được tải tiếp
từ nguồn ModelScope chính thức, có cùng SHA256:
`f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d`.
Các mô hình phụ được lưu trong `models/`.

Script khởi chạy dùng GPU CUDA, địa chỉ 127.0.0.1 và tắt `torch.compile` trên
Windows. Lần tạo âm thanh đầu tiên sau mỗi lần khởi động cần nạp mô hình.
Script cũng chừa 15% VRAM cho Windows, giải phóng bộ nhớ CUDA tạm giữa các lượt
và chạy ASR gốc trên CPU để hạn chế chậm do GPU phải dùng RAM chia sẻ.
Phần đọc văn bản có ba mức **Nhanh / Cân bằng / Ưu tiên giống mẫu**;
xem số đo và cách chọn trong `voice_studio/README.md`.

Đã kiểm tra: 143 gói tương thích bằng `uv pip check`, phép tính BF16 trên RTX 5050,
đọc WAV qua TorchCodec/FFmpeg, giao diện HTTP 200, script tắt/khởi động lại,
và tạo âm thanh tiếng Việt 48 kHz dài 2,56 giây với seed 42 qua API của giao diện.
Kết quả chi tiết: `output/smoke-result.json`.
