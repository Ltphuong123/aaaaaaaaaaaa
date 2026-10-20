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

## Xóa giọng đã tạo

Vào **Thư viện giọng → Của tôi**, bấm **Xóa giọng** trên giọng muốn xóa,
kiểm tra tên trong hộp xác nhận rồi bấm **Xóa giọng**. Chọn **Giữ lại** hoặc nhấn
Escape để hủy. Các giọng **Có sẵn** được giữ trong thư viện.

Khi xóa giọng đang chọn, tool tự chuyển sang một giọng còn lại. Nếu thư viện trống,
nút đọc được tắt cho đến khi có giọng mới. Giọng đang được dùng cho tác vụ đọc
đang chờ/chạy cần đợi tác vụ kết thúc mới xóa được.

Xóa giọng chỉ xóa hồ sơ và mẫu riêng của giọng trong thư viện. Các file WAV đã tạo,
lịch sử đọc và âm thanh tham chiếu đã tải lên vẫn được giữ lại. Có thể lưu lại
một giọng mới với cùng tên sau khi xóa.

Đã kiểm tra API bảo vệ giọng đang dùng/giọng có sẵn, giữ nguyên WAV và lịch sử,
chặn đường dẫn vượt thư viện và junction. Chrome đã kiểm tra hủy/xác nhận,
lỗi xung đột, đổi giọng sau xóa, tải lại trang, thư viện trống, điện thoại và
đọc tiếp bằng giọng còn lại. Các giọng có sẵn trên máy được đối chiếu và giữ nguyên.

## Xóa lịch sử đọc

Trong **Đọc văn bản → Những bản đọc gần đây**, bấm **Xóa** ở bản muốn bỏ,
hoặc **Xóa toàn bộ** để dọn cả các bản cũ không hiển thị trên màn hình.
Kiểm tra hộp xác nhận, rồi bấm **Xóa bản đọc** / **Xóa toàn bộ**; chọn **Giữ lại**
hoặc nhấn Escape để hủy.

Tool xóa nội dung lịch sử cùng WAV và bản âm thanh mẫu của bản đọc tương ứng.
Thư viện giọng, các bản thiết kế giọng, mẫu đã tải lên và văn bản đang soạn vẫn
được giữ lại. Bản đọc bị xóa sẽ không khôi phục khi tải lại trang. Các file đã
tải về riêng ngoài tool vẫn còn trên máy. Đợi tác vụ đọc đang chạy hoàn tất
trước khi xóa toàn bộ lịch sử. Nếu file đang được dùng, dừng phát/tải rồi thử lại;
tool thông báo các bản chưa xóa được.

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

### Tăng tốc ở nguyên chất lượng ưu tiên giống mẫu

Script khởi chạy bật các tối ưu sau, không thay đổi số bước, CFG, seed hay âm thanh mẫu:

- **Tái sử dụng đặc trưng giọng mẫu:** giữ tối đa 16 mục / 16 MiB đặc trưng trên CPU.
  Mẫu không phải mã hóa lại khi đọc đoạn mới. Khóa gồm nội dung file và lời mẫu;
  đổi file hoặc lời mẫu sẽ được xử lý lại. Bộ nhớ đệm hết khi tắt tiến trình.
  Tác vụ bật khử nhiễu vẫn đi qua đầy đủ đường xử lý gốc.
- **CUDA Graph cho bộ ước lượng LocDiT:** dùng lại chuỗi lệnh GPU có kích thước cố định.
  Phần lấy mẫu ngẫu nhiên, mô hình ngôn ngữ và 10 bước suy luận vẫn giữ nguyên.
  Tối ưu tự đối chiếu đầu ra với cách chạy gốc trước khi dùng; nếu không phù hợp thì
  quay lại bộ ước lượng gốc. Không cần cài Triton.
- **CUDA Graph cho các khối sinh từng token:** dùng lại lệnh GPU của Base LM,
  Residual LM và bộ mã hóa một patch âm thanh. Phần xử lý ngữ cảnh ban đầu có độ dài
  thay đổi vẫn chạy theo đường gốc. Bộ đếm vị trí và việc làm sạch KV cache giữa
  các câu vẫn giữ nguyên; mỗi khối được đối chiếu đầu ra chính xác trước khi bật.

Đo ở lượt tối ưu cache giọng/LocDiT trên máy này, cùng giọng Nam trầm ấm, seed 42, CFG 2 và 10 bước,
sau khi đã nạp mô hình (số đo từng lượt, có thể thay đổi theo tải máy):

| Văn bản | Trước hai tối ưu này | Sau tối ưu |
| --- | ---: | ---: |
| Câu ngắn, WAV 3,2 giây | 7,25 giây | 2,46–2,52 giây |
| Câu mới dài hơn, WAV 5,92 giây | 12,49 giây | 4,53 giây |

Đã đối chiếu SHA256: cả bốn lượt WAV thử, gồm nội dung mới, trùng từng byte với bản gốc.
Đã thử đọc dài hai đoạn bằng chế độ **Ưu tiên giống mẫu**: WAV 18,74 giây tạo trong
14,25 giây. Thư viện giọng được giữ nguyên. Tạo giọng mới, khử nhiễu, chuẩn hóa,
seed 0 và seed ngẫu nhiên vẫn qua kiểm tra với mô hình thật.

Lượt tối ưu tiếp theo mở rộng CUDA Graph sang ba khối sinh token, vẫn cùng cấu hình trên:

| Văn bản | Trước mở rộng graph | Sau mở rộng graph |
| --- | ---: | ---: |
| Câu ngắn, WAV 3,2 giây | 2,55 giây | 2,32 giây |
| Câu dài hơn, WAV 5,92 giây | 4,38 giây | 4,07 giây |

Câu ngắn lấy trung bình hai lượt sau khi mô hình đã nạp; câu dài hơn đo một lượt.
Mức cải thiện thêm khoảng 7–9%, phụ thuộc tải máy. Bốn file WAV đối chiếu vẫn
trùng từng byte với trước khi mở rộng graph. Kiểm tra bộ nhớ trạng thái xác nhận
graph dùng đúng vị trí mới và vẫn hoạt động sau khi KV cache được làm sạch.
Đoạn dài hai phần, WAV 18,74 giây, được tạo trong 12,78 giây và cũng trùng từng byte
với bản đọc trước khi mở rộng graph.

Khi bật khử nhiễu, tool tạm nhả workspace của ba graph sinh token để chừa VRAM
cho ZipEnhancer. Chúng sẽ được tạo lại khi đọc tiếp. Nếu phát sinh thiếu VRAM ở
đường tạo thông thường, tool thử lại một lần với các workspace này đã nhả, giữ
nguyên seed và mọi thông số chất lượng.

Các biến trong `start-voxcpm.ps1`: `VOXCPM_PROMPT_CACHE_SIZE=16`,
`VOXCPM_CUDA_ESTIMATOR_GRAPH=1` và `VOXCPM_CUDA_TOKEN_GRAPHS=1`.
Đặt về `0` và khởi động lại backend để tắt từng tối ưu.
Tài liệu kỹ thuật: [prompt cache của VoxCPM](https://github.com/OpenBMB/VoxCPM/blob/main/src/voxcpm/model/voxcpm2.py)
và [CUDA Graphs của PyTorch 2.10](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs).

### Tối ưu thêm cho cả ba chế độ đọc

Các tối ưu bổ sung giữ nguyên bước 4/6/10, CFG, seed, toàn bộ ngữ cảnh,
âm thanh tham chiếu và cách chia đoạn:

- **Graph cho toàn bộ bộ giải CFM:** gom cả các bước Euler/CFG và LocDiT;
  việc lấy nhiễu ngẫu nhiên vẫn nằm ngoài graph. Chỉ giữ một graph CFM để giới hạn
  VRAM; đổi chế độ có thể cần tạo lại graph. Graph kiểm tra kết quả bằng đường
  gốc và tự quay về cách cũ nếu không phù hợp.
- **Dùng lại lưới thời gian:** giữ tối đa tám vector thời gian được tính bằng
  đúng các phép toán gốc trên cùng thiết bị/dtype. Không đổi số bước suy luận.
- **Giảm các điểm chờ CPU/GPU:** tái sử dụng hai bộ đệm vị trí token, chuyển
  cờ kết thúc sang bộ nhớ CPU pinned trước khi xử lý patch và đợi đúng CUDA event
  khi kiểm tra dừng. Giữ nguyên patch cuối và quy tắc độ dài tối thiểu;
  CPU và chế độ streaming tiếp tục dùng đường cũ.
- **Dùng lại kết nối mô hình:** Giọng Nhà tái sử dụng Gradio Client giữa các
  lượt đọc. Khi kết nối lỗi, lượt tiếp theo kết nối lại; không tự sinh hai lần.
- **Dùng lại bản đọc trùng hoàn toàn:** trong cùng phiên chạy, giữ chỉ mục tối đa
  32 bản đã đọc. Cùng văn bản, mẫu giọng, lời mẫu, seed và thiết lập sẽ dùng nguyên
  WAV đã tạo, lưu thành bản lịch sử mới và hiện **Dùng lại bản đọc đã tạo**.
  Thay mẫu/cấu hình/mô hình hoặc khởi động lại backend sẽ tạo lại âm thanh.
  Không dùng cách này cho tạo giọng, seed ngẫu nhiên, khử nhiễu hoặc backend từ xa.
  Chỉ mục không giữ thêm file âm thanh ẩn: xóa lịch sử sẽ xóa các nguồn tương ứng;
  nếu nguồn đã xóa hay bị sửa, lượt đọc sau chạy mô hình bình thường.
- **Hiện kết quả sớm hơn:** kiểm tra tiến độ mỗi 250 ms thay vì 1.200 ms.
  Việc này giảm độ trễ giao diện khoảng 0,475 giây trung bình, không thay đổi
  tốc độ tính toán của mô hình.
- **Nhả bộ nhớ khử nhiễu:** sau khi dùng ZipEnhancer, chuyển trọng số và bộ đệm
  vị trí của nó về RAM; lần khử nhiễu tiếp theo chuyển lại cùng thiết bị GPU.
  Việc này giữ chỗ cho đọc dài, hạn chế phải chạy lại do thiếu VRAM sau khi tạo giọng.

Script bật `VOXCPM_CUDA_CFM_GRAPH=1`, `VOXCPM_ASYNC_INFERENCE=1`
và `VOXCPM_OFFLOAD_DENOISER=1`.
Đặt từng biến về `0` rồi khởi động lại backend để tắt. Không cần thư viện mới.
Lần đầu nạp mô hình hoặc khởi tạo graph vẫn lâu hơn các lượt đã làm nóng.

Đo ngày 21/10/2026 trên RTX 5050 8 GB, giọng Nam trầm ấm, seed 42,
câu ngắn ở bảng trước. Trung bình hai lượt đã làm nóng mỗi chế độ, đo qua
Gradio Client dùng lại kết nối, không dùng lại WAV đã tạo:

| Chế độ | Trước lượt tối ưu này | Sau |
| --- | ---: | ---: |
| Nhanh, 4 bước | 1,806 giây | 1,776 giây |
| Cân bằng, 6 bước | 2,077 giây | 2,017 giây |
| Ưu tiên giống mẫu, 10 bước | 2,385 giây | 2,256 giây |

Mức tăng thêm nhỏ và có thể nằm gần biến động tải máy, đặc biệt ở chế độ Nhanh.
Đây là thời gian sinh/tải WAV, chưa cộng thời gian đợi lượt polling giao diện.
12 WAV đối chiếu của ba chế độ với hai nội dung trùng từng byte;
ba lượt giọng nữ và một lượt seed 0 cũng trùng WAV với bản trước.
Số đo chi tiết cục bộ: `.runtime/speed-final-comparison.json`.

Với cùng nội dung/giọng/thiết lập được đọc lại ngay trong cùng phiên, cả ba chế độ
đo được khoảng **0,01 giây xử lý** nhờ dùng nguyên WAV trước; kiểm tra xác nhận
không gọi mô hình và tạo lịch sử mới với ID riêng. Đây là dùng lại âm thanh,
không phải tốc độ tạo một nội dung mới. Chi tiết: `.runtime/speed-pipeline-report.json`.

Đã kiểm tra chuỗi khử nhiễu hai lần, thiết kế giọng rồi đọc dài ở 10 bước.
Trước khi nhả trọng số khử nhiễu, bản đọc 18,74 giây mất 41,8 giây vì phải thử lại
do thiếu VRAM. Sau khi nhả, cùng WAV được tạo trong 12,34 giây, không phải thử lại.
Cả hai lượt khử nhiễu và bản đọc dài đều trùng từng byte với WAV chuẩn.
58 bài kiểm tra tự động đã qua; Chrome kiểm tra ba chế độ, trạng thái dùng lại
bản đọc, polling 250 ms và bố cục máy tính/điện thoại. Thử nghiệm dùng dữ liệu
riêng và đối chiếu hash để giữ nguyên thư viện giọng cùng lịch sử của người dùng.

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
.\.venv\Scripts\python.exe -m unittest discover -s voice_studio -p "test_*.py" -v
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
