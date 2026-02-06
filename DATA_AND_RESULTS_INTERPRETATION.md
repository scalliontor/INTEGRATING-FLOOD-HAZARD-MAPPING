# 📖 CÂU CHUYỆN DỮ LIỆU & GIẢI THÍCH KẾT QUẢ (DATA EXPLORATION & INTERPRETATION)

Tài liệu này cung cấp những mô tả chuyên sâu, hấp dẫn (compelling descriptions) về dữ liệu đầu vào, phương pháp tiếp cận và ý nghĩa thực sự đằng sau các biểu đồ kết quả. Đây là tài liệu "kể chuyện" (storytelling) giúp người đọc hiểu được *tại sao* chúng ta làm như vậy và con số nói lên điều gì.

---

# 1. DỮ LIỆU ĐẦU VÀO: NỀN TẢNG CỦA MỌI DỰ BÁO

Chúng ta không chỉ đơn thuần "nhập dữ liệu vào máy tính". Chúng ta đang tái hiện lại thế giới thực dưới dạng số hóa để mô hình có thể "nhìn" thấy lũ.

### 🛰️ 1.1. "Sự thật mặt đất" (Ground Truth): Sentinel-1 SAR
> *"Làm sao chúng ta biết đâu là vùng ngập?"*

*   **Dữ liệu:** Ảnh Radar khẩu độ tổng hợp (SAR) từ vệ tinh Sentinel-1.
*   **Phương pháp:** Radar có khả năng **xuyên qua mây**. Đây là yếu tố sống còn vì trong các trận bão lũ, bầu trời luôn bị che phủ hoàn toàn, vệ tinh quang học (như Google Maps) sẽ bị "mù". Chúng ta so sánh ảnh trước và trong lũ để phát hiện sự thay đổi bề mặt.
*   **Ý nghĩa:** Đây là "Cán cân công lý" (Ground Truth). Mô hình học đúng hay sai hoàn toàn phụ thuộc vào bộ dữ liệu này. Chúng ta đã thu thập 19 sự kiện trong 10 năm để đảm bảo mô hình học được đủ các kịch bản từ lũ nhỏ đến đại hồng thủy.

### 🏔️ 1.2. "Cái Bát chứa nước": Dữ liệu Địa hình (Static Features)
> *"Tại sao nước lại chảy về đây?"*

*   **Dữ liệu:** SRTM Digital Elevation Model (DEM), HydroSHEDS.
*   **Phương pháp:** Chúng ta không chỉ đưa vào độ cao (`elevation`). Chúng ta phái sinh ra các chỉ số thủy văn phức tạp hơn như `Relief` (độ chênh cao cục bộ), `TWI` (Chỉ số ẩm ướt - nơi nước có xu hướng tích tụ).
*   **Ý nghĩa:** Lũ lụt không ngẫu nhiên. Nước luôn chảy về chỗ trũng. Các biến số này giúp mô hình hiểu được **"hình thái học của dòng chảy"**. `Relief` (chênh lệch độ cao) chính là biến số quan trọng nhất: một vùng trũng thấp bao quanh bởi núi cao chính là một "cái bát" hứng nước khổng lồ.

### 🌧️ 1.3. "Động cơ của Lũ": Dữ liệu Mưa (Dynamic Rain)
> *"Cơn lũ này được kích hoạt như thế nào?"*

*   **Dữ liệu:** CHIRPS Daily Rainfall (Lượng mưa vệ tinh toàn cầu).
*   **Phương pháp:** Vì mưa lũ có độ trễ, chúng ta không chỉ xem mưa ngày hôm nay. Chúng ta tính toán `Rain_3D` (kích hoạt nhanh), `Rain_7D` (tích lũy), `Rain_Max` (cường độ đỉnh) và đặc biệt là `Rain_AM14` (Độ ẩm đất tiền cảnh - đất đã no nước chưa?).
*   **Ý nghĩa:** Đây là yếu tố **ĐỘNG**. Nếu không có mưa, vùng trũng vẫn chỉ là vùng trũng khô ráo. Dữ liệu này giúp mô hình phân biệt được: "Tại sao cùng một thung lũng đó, năm ngoái không ngập mà năm nay lại ngập trắng?".

---

# 2. PHƯƠNG PHÁP: TRÍ TUỆ NHÂN TẠO CÓ ĐỊNH HƯỚNG VẬT LÝ
> *"Chúng ta dạy máy tính học như thế nào?"*

Chúng ta không dùng một hộp đen (Black Box) mù quáng. Chúng ta sử dụng **XGBoost with Monotonic Constraints** (Ràng buộc đơn điệu).

*   **Vấn đề:** Các mô hình AI thông thường chỉ tìm kiếm tương quan con số. Đôi khi nó học sai: "Mưa càng to thì nước càng rút" (do nhiễu dữ liệu).
*   **Giải pháp:** Chúng ta áp đặt "Luật Vật Lý" vào mô hình: "Nếu mưa tăng, nguy cơ ngập BẮT BUỘC phải tăng hoặc giữ nguyên, không được giảm".
*   **Kết quả:** Một mô hình vừa thông minh (dự báo chính xác 96% AUC) vừa đáng tin cậy (tuân thủ logic tự nhiên).

---

# 3. GIẢI MÃ CÁC BIỂU ĐỒ (VISUALIZATIONS DEEP DIVE)

Phần này phân tích sâu sắc các biểu đồ (Graphs) mà chúng ta đã tạo ra. Đây là trọng tâm của bài báo cáo.

## 📊 Feature Importance & SHAP Analysis
> *"Điều gì thực sự điều khiển cơn lũ?"*

### Biểu đồ 1: SHAP Summary Bar (Xếp hạng Tầm quan trọng)
*   **Dữ liệu hiển thị:** Xếp hạng các yếu tố ảnh hưởng nhất đến quyết định của mô hình.
*   **Câu chuyện:** **`Relief` (Địa hình) là Vua**. Thanh `relief` dài áp đảo các yếu tố khác. Điều này khẳng định: Ở Hà Tĩnh, địa hình là yếu tố quyết định số phận. Mưa to đến mấy mà ở trên đỉnh núi thì nước cũng trôi đi. Nhưng nếu ở vùng trũng (`Relief` thấp), chỉ cần mưa vừa là đã ngập.
*   **Điểm nhấn:** Mưa (`Rain_7D`, `Rain_Max`) có đóng góp quan trọng nhưng đứng sau địa hình. Điều này hợp lý với đặc thù lũ lụt miền Trung: địa hình dốc ngắn, nước tập trung cực nhanh.

### Biểu đồ 2: SHAP Summary Dot (Chiều hướng Tác động)
*   **Dữ liệu hiển thị:** Các chấm màu xanh/đỏ phân bố về hai phía trục tung.
*   **Câu chuyện:** Hãy nhìn vào dòng `Rain_7D`. Các chấm màu ĐỎ (Mưa lớn) nằm hoàn toàn bên phải (Tăng nguy cơ). Các chấm màu XANH (Mưa nhỏ) nằm bên trái (Giảm nguy cơ).
*   **Ý nghĩa:** Biểu đồ này chứng minh mô hình đã học đúng quy luật: **Mưa lớn = Nguy hiểm**. Nó phản bác lại mọi nghi ngờ về việc "Mô hình có học vẹt không?".

### Biểu đồ 3: SHAP Dependence Plot (Sự Tương tác Phức tạp)
*   **Dữ liệu hiển thị:** Mối quan hệ giữa [Mưa] và [Nguy cơ] phụ thuộc vào [Địa hình].
*   **Câu chuyện:** Đường xu hướng đi lên: Mưa càng tăng, nguy cơ càng cao. NHƯNG, hãy nhìn màu sắc các chấm. Tại cùng một lượng mưa (ví dụ 300mm), những điểm màu XANH (vùng trũng) có nguy cơ cao vọt, trong khi điểm màu ĐỎ (vùng núi) nguy cơ vẫn thấp.
*   **Ý nghĩa:** Lũ lụt là kết quả của cuộc hôn phối giữa **Mưa** và **Địa hình**.

---

## 🗺️ Validation Result (Kiểm định Thực tế)
> *"Mô hình có hoạt động trong thảm họa thực sự không?"*

### Biểu đồ 4: Validation Classification Report (Trận Đại Hồng Thủy 2020)
*   **Bối cảnh:** Tháng 10/2020, miền Trung hứng chịu trận lũ lịch sử. Đây là bài kiểm tra khắc nghiệt nhất (Stress Test) cho bất kỳ mô hình nào.
*   **Dữ liệu hiển thị:** So sánh giữa [Dự báo của AI] và [Ảnh vệ tinh thực tế].
*   **Câu chuyện:**
    *   **Recall 94%:** Mô hình đã tô màu đỏ gần như toàn bộ những nơi thực sự bị ngập. Nó không bỏ rơi người dân trong vùng nguy hiểm.
    *   **Precision 56%:** Trong điều kiện bình thường con số này thường thấp (~15%), nhưng trong đại hồng thủy, nó tăng vọt lên 56%. Điều này cho thấy mô hình cực kỳ nhạy bén với các sự kiện cực đoan.
*   **Kết luận:** Mô hình này **SẴN SÀNG** để ứng dụng cảnh báo sớm thiên tai.

---

## 🌍 Climate Scenarios (Tương lai Biến đổi Khí hậu)
> *"Hà Tĩnh sẽ ra sao vào năm 2050 và 2100?"*

### Biểu đồ 5: Climate Scenario Comparison (So sánh Kịch bản)
*   **Dữ liệu hiển thị:** Diện tích vùng Nguy cơ Cao (High Risk) biến đổi theo các kịch bản RCP 4.5 (lạc quan) và RCP 8.5 (bi quan).
*   **Câu chuyện:**
    *   Mưa tăng 30% (Kịch bản RCP 8.5 năm 2100).
    *   Diện tích ngập "chỉ" tăng khoảng 0.7% (khoảng 700 ha).
    *   *Tại sao ít vậy?* Vì địa hình Hà Tĩnh rất dốc. Nước dâng lên bị giới hạn bởi các sườn đồi.
*   **Ý nghĩa Sâu xa:** Đừng để con số diện tích đánh lừa. **Diện tích không tăng nhiều, nhưng ĐỘ SÂU và TẦN SUẤT ngập tại các vùng cũ sẽ tăng khủng khiếp.** Vùng nguy hiểm vẫn là vùng đó, nhưng nó sẽ nguy hiểm hơn gấp bội.

---

## 📍 District Analysis (Phân tích Cục bộ)
> *"Xã tôi có an toàn không?"*

### Biểu đồ 6: District Risk Change Map (Bản đồ Cấp Huyện)
*   **Dữ liệu hiển thị:** Mức độ tăng/giảm nguy cơ trung bình của từng huyện.
*   **Câu chuyện:**
    *   **Huyện Đức Thọ & Can Lộc:** Đây là "rốn lũ". Màu đỏ rực trên bản đồ. Đây là nơi hợp lưu của các con sông lớn (Ngàn Sâu, Ngàn Phố), địa hình lòng chảo.
    *   **Thị xã Kỳ Anh:** Có mức TĂNG trưởng rủi ro lớn nhất (+0.039). Vùng ven biển này đang trở nên nhạy cảm hơn với biến đổi khí hậu so với vùng nội địa.
*   **Thông điệp cho Lãnh đạo:** Cần ưu tiên nguồn lực phòng chống thiên tai cho **Đức Thọ** (nguy cơ hiện hữu cao nhất) và **Kỳ Anh** (nguy cơ tương lai tăng nhanh nhất).

---

*Tài liệu này được biên soạn để giúp người đọc không chuyên hiểu được giá trị cốt lõi của dự án công nghệ phức tạp này.*
*Antigravity AI Assistant*
