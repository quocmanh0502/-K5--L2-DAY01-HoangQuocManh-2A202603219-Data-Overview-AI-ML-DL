# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…

`class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

(ảnh `traffic`, `coco_image_id`=210273, kích thước 640 x 428, model `yolo11n-cls.pt`)

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?

Record này nói về **cả bức ảnh** vì:
- checkpoint `yolo11n-cls.pt` là model phân loại, nhìn toàn bộ ảnh rồi trả ra một danh sách top-5 lớp cho cả ảnh (mỗi ảnh đúng 5 dòng rank 1→5)
- trong JSON không có trường nào chỉ vị trí vật thể (không có `bbox_xyxy`, không có `polygon_xy`), chỉ có `image_width`/`image_height` của cả ảnh
- ảnh traffic có rất nhiều xe (bus, car, người…) nhưng model vẫn chỉ trả ra một nhãn hạng 1 là `cab` → nhãn đại diện cho cả cảnh, không gắn với chiếc xe cụ thể nào

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

Class list **đi kèm taxonomy/checkpoint**, model không tự nghĩ ra:
- `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên có đúng 1000 lớp cố định, đã được định nghĩa sẵn khi xây dựng bộ dữ liệu
- model chỉ chọn trong 1000 lớp đó; nếu vật trong ảnh không có trong danh sách thì model vẫn phải chọn lớp gần nhất (vd. ảnh xe buýt nhưng ra `cab`, `minibus`)

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

- Tên lớp có thể trùng hoặc khác nghĩa giữa các taxonomy. Ví dụ `cab` ở ImageNet-1K là id 468, còn COCO-80 không có lớp `cab` mà chỉ có `car` (id 2); cùng id 2 ở hai taxonomy lại là hai lớp khác nhau
- `class_id` chỉ có nghĩa khi biết nó thuộc taxonomy nào → ghi đủ 3 trường thì nhãn rõ ràng, tra lại được
- Lưu đủ 3 trường giúp việc chuyển đổi (migration) hoặc ánh xạ giữa các phiên bản dữ liệu/taxonomy cũ và mới không bị mất thông tin

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

- chọn đối tượng **chính**: vật chiếm diện tích lớn nhất / ở tiền cảnh / rõ nét nhất
- với ảnh góc rộng nhiều vật (như traffic) thì ưu tiên lớp mô tả cả cảnh hoặc theo thứ tự ưu tiên lớp quy định sẵn trong guideline (vd. bus > car > person)
- chỉ được chọn lớp **có trong taxonomy**, không tự đặt tên mới
- nếu hai lớp ngang nhau, không quyết định được → escalation hỏi Lab Coach, không tự bịa

6. Vì sao không được lấy `score` của model làm ground truth?

- `score` chỉ là mức “tự tin” của model, không phải điểm đúng/sai. Model có thể sai mà vẫn tự tin: ảnh traffic chủ yếu là xe buýt nhưng top-1 là `cab` (0.51)
- ground truth phải do con người gán theo guideline; nếu lấy prediction làm nhãn thì lỗi của model sẽ bị đưa ngược lại vào dữ liệu huấn luyện

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…

`class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

(ảnh `kitchen`, 640 x 427, `class_id`=0, `taxonomy_name`=COCO-80)

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

- gốc (0, 0) là góc trên-trái của ảnh, x tăng sang phải, y tăng xuống dưới
- góc trên-trái của box ở (385.33, 69.24): cách mép trái ảnh ~385 px, cách mép trên ~69 px
- góc dưới-phải ở (498.92, 348.92)
- box rộng 498.92 − 385.33 ≈ 113.58 px, cao 348.92 − 69.24 ≈ 279.68 px → hộp đứng, cao gần gấp 2.5 lần chiều rộng, đúng dáng người đàn ông đứng quay lưng ở giữa-phải ảnh

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

| Ngưỡng | Số prediction (ảnh kitchen) |
| --- | --- |
| 0.20 | 17 |
| 0.35 (dùng cho JSON) | 11 |
| 0.60 | 6 (person, bowl, bowl, oven, oven, person) |

→ hạ ngưỡng từ 0.60 xuống 0.20 thì số hộp tăng gần gấp 3; các hộp thêm vào chủ yếu là bowl/cup có score thấp.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

- kiểm tra nhiều hộp hơn, phần lớn là hộp score thấp → dễ sai
- loại hộp không có vật thật (false positive), vd. `bowl` 0.38 ở [0.32, 305.02, 87.69, 371.32] chồng lấn với `bowl` 0.72 ngay bên cạnh → có thể là hộp trùng
- kiểm tra hộp trùng nhau cho cùng một vật, hộp sai lớp (cup hay bowl?), hộp lệch biên
- ngược lại, khi tăng ngưỡng thì phải tìm vật bị **bỏ sót** (vd. ở 0.60 mất hết cup)
- Lưu ý: threshold chỉ lọc prediction, không phải quy tắc bỏ object khi làm ground truth

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

**Quy tắc box:** Mỗi vật thể thuộc taxonomy COCO-80 được vẽ **đúng một** hộp chữ nhật bao **sát phần nhìn thấy** của vật (không chừa khoảng trống, không cắt mất phần vật); không gộp hai vật vào một hộp; vật bị che hoặc bị cắt ở mép ảnh vẫn gán nếu nhận ra được lớp, và chỉ vẽ phần nằm trong ảnh.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

- Vật: hộp `person` score 0.61 ở mép trái ảnh, `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78] (x_min ≈ 0 → bị cắt ở mép trái), hộp nhỏ chỉ 61 x 48 px, nằm lẫn với nồi/bát trên bàn bếp. Model segmentation cũng chỉ cho score 0.43 cho vật này
- Cách xử lý:
  - phần **“vật bị cắt mép thì vẽ phần nhìn thấy”** → viết thành **quy tắc trong guideline** (đã có ở câu 5), vì đây là trường hợp lặp lại ở rất nhiều ảnh
  - nhưng riêng vật này nhìn trên PNG **không chắc có phải người hay không** (chỉ thấy một mảng nhỏ ở mép) → **escalation** hỏi Lab Coach, vì guideline không thể quyết định thay khi chính lớp của vật còn mơ hồ; không tự bịa nhãn
- Thêm: trên bàn có một **chai** (bottle) màu nâu ở giữa ảnh nhưng model không phát hiện → vật bị bỏ sót, annotator vẫn phải gán

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…

`instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[[446, 70], [445, 71], [444, 71], [443, 72], [442, 72]]

(`bbox_xyxy`=[385.45, 66.44, 498.02, 348.58])

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

- box chỉ có 4 số → hình chữ nhật, bao gồm cả nền xung quanh (tường, nồi treo phía sau người)
- polygon có 348 điểm bám theo đường viền → biết **hình dạng thật** của người: đầu, vai, tay, chân, khoảng trống giữa hai chân
- từ polygon tính được diện tích thật của vật, biết pixel nào thuộc vật/không thuộc vật, tách được vật khi hai vật chồng lên nhau

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

- `instance_id` dùng để **phân biệt từng vật riêng lẻ** trong output của lab (kitchen-001, kitchen-002…), mỗi vật một polygon riêng
- **không phải class ID**: nhiều instance có thể cùng lớp, vd. `kitchen-001` và `kitchen-009` đều là `person` (class_id 0) nhưng là hai instance khác nhau; `kitchen-002`, `-003`, `-006`, `-010` đều là `bowl`
- **không phải tracking ID**: nó chỉ là số thứ tự trong output của một ảnh, đánh lại mỗi lần chạy; không dùng để theo dõi cùng một vật qua nhiều ảnh/frame video

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

**Quy tắc mask:** Đường viền polygon phải bám sát mép ngoài của phần **nhìn thấy** của vật, không lấn sang nền hay vật bên cạnh; phần bị vật khác che thì **không** vẽ; hai vật chạm nhau phải là hai polygon riêng, ranh giới đặt theo vật ở phía trước (vật trước giữ trọn biên); lỗ hở thật của vật (vd. khoảng giữa hai chân) không tô vào mask.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

- Chỗ: mask `kitchen-005` **dining table** (score 0.578, 598 điểm, bbox [0.97, 234.74, 345.86, 423.87]) phủ gần hết nửa dưới bên trái ảnh, chồng lên các bát `kitchen-002`, `kitchen-003`, `kitchen-006` đặt trên bàn → hai vật chạm/đè nhau, khó biết biên bàn ở đâu
- Dùng **guideline**: quy tắc ở câu 4 đã giải quyết được (bát ở phía trước giữ trọn biên, phần bàn bị bát che thì không tô vào mask bàn). Đây là trường hợp rất phổ biến (đồ vật trên bàn) nên phải có quy tắc chung, không cần hỏi từng lần
- Ngoài ra `kitchen-004` **potted plant** 0.63 ở góc trên-trái ([0.05, 1.4, 63.45, 148.89]) trông giống đồ treo/rổ trên tường hơn là chậu cây → nếu nhìn ảnh vẫn không chắc vật đó là gì thì **escalation**, không tự gán lớp

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **1 nhãn cho cả ảnh**: `class_id` + `class_name` + `taxonomy_name` (vd. ImageNet-1K) | `traffic` toàn xe buýt/ô tô nhưng top-1 là `cab` (0.51), top-2 `minibus` (0.16) → ảnh nhiều vật, khó chọn 1 lớp, taxonomy ImageNet không khớp cảnh | Xem cả ảnh, chọn 1 lớp chính trong taxonomy theo quy tắc ưu tiên; ca mơ hồ → escalation | Lớp có đúng vật chính không, có nằm trong taxonomy không, id/tên/taxonomy có khớp nhau không |
| Phát hiện vật thể | **1 box `bbox_xyxy` (pixel) + lớp COCO-80 cho mỗi vật** | `kitchen`: `person` 0.61 ở mép trái [0.12, 263.18, 61.55, 310.78] không chắc là người; `bowl` 0.38 chồng lên `bowl` 0.72; chai trên bàn bị bỏ sót; số hộp đổi 17 → 11 → 6 theo ngưỡng 0.20/0.35/0.60 | Vẽ 1 box sát phần nhìn thấy cho mọi vật trong taxonomy (kể cả vật model bỏ sót), gán đúng lớp | Box thiếu/thừa/trùng, box lệch biên, sai lớp, vật bị che/cắt mép có xử lý đúng guideline không |
| Instance segmentation | **1 polygon `polygon_xy` + lớp + `instance_id` riêng cho mỗi vật** | `kitchen-005` dining table phủ đè lên các bát `kitchen-002/003/006`; `kitchen-004` potted plant 0.63 ở góc trên-trái có thể sai lớp; `kitchen-009` person chỉ 0.43 | Vẽ viền sát mép phần nhìn thấy, tách mỗi vật thành 1 polygon, không tô phần bị che | Viền lấn nền/lấn vật khác, gộp nhầm 2 vật, thiếu instance, `instance_id` không trùng, lớp đúng chưa |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

Chỉ dùng 3 ảnh COCO công khai (traffic, kitchen, dining) do notebook tải về và kiểm tra SHA-256, không tải ảnh cá nhân/dữ liệu nội bộ, và không ghi họ tên, MSSV, email hay dữ liệu cá nhân nào vào báo cáo/output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

Dừng ngay, không gán nhãn và không chia sẻ dữ liệu đó, rồi báo cho **Lab Coach / giảng viên phụ trách** (mentor) để được hướng dẫn.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
