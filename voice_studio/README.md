# Giọng Nhà

Tool cục bộ dùng VoxCPM2 để tạo giọng, lưu giọng và đọc văn bản bằng giọng đã chọn.

## Mở tool

Nhấp đúp `start-voice-studio.cmd` ở thư mục VoxCPM, sau đó mở **http://localhost:8809**.
Script tự chạy VoxCPM ở cổng 8808 nếu cần. Không chạy thêm một bản mô hình trên GPU.
Nhấp đúp `stop-voice-studio.cmd` để tắt cả tool và mô hình, giải phóng GPU.

## Cách dùng

1. **Đọc văn bản:** chọn giọng, nhập nội dung và bấm **Tạo giọng đọc**. Nghe kết quả hoặc tải WAV.
2. **Thư viện giọng:** nghe thử, tìm kiếm, lọc giọng **Có sẵn** / **Của tôi**, rồi bấm **Dùng giọng này**.
3. **Tạo giọng mới:** dùng các chế độ của VoxCPM2 bên dưới, nghe thử, đặt tên rồi **Lưu vào thư viện**.

## Tốc độ đọc văn bản

Màn hình đọc có ba lựa chọn, được nhớ cho lần mở sau:

- **Nhanh:** 4 bước LocDiT, dùng âm thanh tham chiếu; ngữ điệu và độ giống mẫu có thể giảm.
- **Cân bằng** (mặc định): 6 bước, dùng cả âm thanh và lời mẫu để giữ chi tiết giọng.
- **Ưu tiên giống mẫu:** 10 bước, cùng chế độ sao chép chi tiết như trước.

Các mức này chỉ áp dụng cho **Đọc văn bản**. Phần **Tạo giọng mới** vẫn dùng đúng
CFG, số bước và các tùy chọn bạn đặt trong cài đặt nâng cao.
Khi đang chạy, tool hiển thị số giây đã chờ; kết quả ghi thời gian xử lý thực tế.
Đây là thời gian tạo âm thanh, không phải tốc độ phát hay thời lượng file WAV.

Script khởi chạy giới hạn bộ nhớ PyTorch ở 85% VRAM, giải phóng bộ nhớ CUDA tạm
giữa các lượt tạo và cho ASR của giao diện gốc chạy CPU. Cách này dành khoảng trống
cho Windows, hạn chế việc driver chuyển dữ liệu GPU sang RAM chia sẻ gây chậm.
Windows hiện chạy PyTorch ở chế độ thường vì môi trường chưa có Triton.
Lần đọc đầu sau khi bật tool cần thêm thời gian nạp mô hình. Văn bản càng dài
và mẫu giọng càng dài thì thường cần nhiều thời gian hơn.

Đo trên RTX 5050 8 GB của máy này, câu “Xin chào, chúc bạn một ngày thật bình yên.”,
giọng Nam trầm ấm, seed 42, sau khi mô hình đã nạp:

| Thiết lập | Trước tối ưu bộ nhớ | Sau tối ưu bộ nhớ |
| --- | ---: | ---: |
| Sao chép chi tiết, 10 bước | 25,35 giây | 5,74 giây |
| Cân bằng: sao chép chi tiết, 6 bước | 22,46 giây | 4,22 giây |
| Nhanh: tham chiếu âm thanh, 4 bước | 18,35 giây | 3,18 giây |

Đây là các lượt đo riêng lẻ, tốc độ thực tế thay đổi theo câu, mẫu và tải máy.
Lượt đầu sau khởi động cần thêm khoảng một phút nạp mô hình, không tính vào bảng.
Đã kiểm tra đọc văn bản dài thành hai đoạn: WAV 19,06 giây tạo trong 24,01 giây;
thư viện giọng không đổi, tải WAV và giao diện điện thoại hoạt động.

## Tạo giọng với các chức năng của VoxCPM gốc

- **Thiết kế giọng:** không thêm âm thanh mẫu. Có thể nhập chỉ dẫn về giới tính, tuổi,
  âm sắc, cảm xúc và nhịp đọc, hoặc để mô hình tự chọn. Có các chỉ dẫn mẫu tiếng Anh/Trung.
- **Sao chép có điều khiển:** tải file hoặc thu âm microphone, rồi nhập chỉ dẫn phong cách nếu cần.
- **Sao chép chi tiết — Ultimate Cloning:** thêm âm thanh, bật chế độ chi tiết và nhập đúng
  lời của âm thanh mẫu. Chỉ dẫn điều khiển được ẩn trong chế độ này, giống cách VoxCPM gốc hoạt động.

Khi bật chế độ chi tiết, lời mẫu được nhận dạng tự động. Có thể chọn ngôn ngữ, nhận dạng lại
hoặc tự nhập/sửa. Hãy nghe và kiểm tra bản chép lời trước khi tạo. Nhận dạng dùng
Faster Whisper base trên CPU, hỗ trợ tiếng Việt và không chiếm thêm VRAM của mô hình TTS.

**Cài đặt nâng cao** gồm khử nhiễu mẫu bằng ZipEnhancer, chuẩn hóa văn bản bằng WeText,
CFG từ 1 đến 3, số bước LocDiT từ 1 đến 50, seed ngẫu nhiên hoặc seed cố định.
Seed bằng 0 hợp lệ. Seed thực tế được trả về và cập nhật sau mỗi lần tạo.
WeText của VoxCPM ưu tiên tiếng Anh/Trung; có thể để tắt khi đọc tiếng Việt.

Âm thanh mẫu hỗ trợ WAV, MP3, M4A, OGG, FLAC, WebM và các định dạng âm thanh phổ biến khác,
tối đa 30 MB, từ 0,5 đến 60 giây. Đoạn khoảng 5–15 giây tiện để dùng làm mẫu.
Microphone yêu cầu quyền của trình duyệt; bản thu được xử lý như file tải lên.
Văn bản tạo thử tối đa 6.000 ký tự. Nếu chia nhiều đoạn, tool dùng giọng của đoạn đầu
làm tham chiếu cho các đoạn sau. Mẫu lưu vào thư viện chỉ lấy đoạn đầu cùng đúng lời của đoạn đó.

Âm thanh mẫu tải/thu và bản chép lời được lưu trong `data/voice_studio/references/`.
Cấu hình tạo giọng và liên kết đến mẫu nguồn được lưu cùng hồ sơ giọng. Các lựa chọn đang chỉnh
trên biểu mẫu được giữ trong trình duyệt khi tải lại trang.

Ba giọng mẫu tiếng Việt đã tạo cho tool: **Nữ dịu dàng**, **Nam trầm ấm**, **Nữ tươi sáng**.
Các tên này mô tả yêu cầu tạo giọng. Hãy nghe mẫu để chọn giọng phù hợp.
Giọng **Giọng kể chuyện** được tạo và lưu khi kiểm tra toàn bộ chức năng.
Các giọng mẫu là âm thanh AI do VoxCPM2 tạo, không phải bộ giọng đặt tên có sẵn của nhà phát hành.

## Giữ và sao lưu giọng

Mỗi giọng có một thư mục trong `data/voice_studio/voices/`, chứa:

- `reference.wav`: đoạn giọng mẫu riêng, không phụ thuộc file nghe thử tạm.
- `voice.json`: tên, mô tả, nội dung mẫu, seed thực tế, cấu hình tạo giọng và thông tin âm thanh.

Tool dùng mẫu cùng nội dung mẫu để đọc văn bản mới bằng giọng đã lưu.
Đóng trình duyệt, xóa dữ liệu trình duyệt hoặc khởi động lại tool không xóa thư viện này.
Để sao lưu, chép toàn bộ `data/voice_studio/`; để khôi phục, chép lại thư mục đó khi tool đã tắt.
Mỗi máy cần cài mô hình VoxCPM2 để đọc tiếp bằng các giọng này.

Các bản đọc và lịch sử nằm trong `data/voice_studio/renders/`. Văn bản dài tối đa 6.000 ký tự
được chia thành các đoạn, giữ nguyên từ ngữ và ghép lại với khoảng nghỉ ngắn.
Nhật ký nằm trong `.runtime/voice-studio.log` và `.runtime/voice-studio-error.log`.

## Kiểm tra

Đã kiểm tra lưu giọng độc lập với file nghe thử, mở lại thư viện từ tiến trình mới,
chia văn bản dài không mất từ, và đường dẫn dữ liệu. Các kiểm tra này chạy bằng:

```powershell
.\.venv\Scripts\python.exe -m unittest voice_studio.test_storage voice_studio.test_creation voice_studio.test_reading -v
```

Chrome đã kiểm tra tạo và lưu một giọng mới, dùng giọng đã lưu đọc nội dung khác,
tải WAV, giữ lựa chọn sau tải lại trang, tìm kiếm/lọc/chọn giọng và bố cục điện thoại.
File minh họa: `output/voice-studio-demo.wav`.

Đã khởi động lại riêng tool và xác nhận thư viện vẫn đủ bốn giọng. Sau đó đã dùng
giọng **Nam trầm ấm** đọc văn bản dài thành hai đoạn, ghép thành WAV 48 kHz dài
19,22 giây. File kiểm tra: `output/voice-studio-preset-demo.wav`.

Phần tạo giọng đã được kiểm tra bằng mô hình thật ở cả ba chế độ, gồm CFG/số bước,
seed cố định bằng 0, seed ngẫu nhiên, khử nhiễu, chuẩn hóa và nhận dạng lời mẫu tiếng Việt.
Chrome đã kiểm tra tải mẫu, tự điền/sửa lời mẫu, giữ bản nháp khi tải lại, tạo và lưu giọng
chi tiết rồi dùng lại để đọc văn bản khác, thu âm WebM bằng microphone giả,
từ chối file hỏng mà không mất mẫu đang dùng, và giao diện điện thoại.

## Bộ nhận dạng lời mẫu

Thư viện bổ sung nằm trong `voice_studio/requirements.txt`, đã được cài trong `.venv`.
Mô hình ASR cục bộ nằm trong `models/whisper-base/`, cần giữ đủ `config.json`, `model.bin`,
`tokenizer.json` và `vocabulary.txt`. Nhận dạng và tạo giọng sau khi cài xong đều chạy trên máy.
