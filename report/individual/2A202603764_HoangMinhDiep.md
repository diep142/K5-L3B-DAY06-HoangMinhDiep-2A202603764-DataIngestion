# Báo cáo cá nhân — Day 6

## Dữ liệu và pipeline
- **Chủ đề/nguồn/quyền:** *Alice's Adventures in Wonderland* — Lewis Carroll, Project Gutenberg ebook #11 ([trang nguồn](https://www.gutenberg.org/ebooks/11), [manifest](../../starter/data/alice_manifest.json)). Manifest ghi ebook miễn phí sử dụng tại Hoa Kỳ theo Project Gutenberg License; cần kiểm tra luật địa phương trước khi phân phối ngoài Hoa Kỳ. Corpus text không có số trang in gốc; trường `page` khi truy xuất dùng `unit_id`.
- **Corpus/chuẩn hóa:** 46 units theo thứ tự nguồn, tổng 142,892 ký tự sau normalization, unit ngắn nhất 1,590 ký tự. Script chuẩn bị nhóm paragraph thô với mục tiêu 3,000 ký tự/unit. 
- **Model/store/chi phí:** `sentence-transformers/all-MiniLM-L6-v2`, FastEmbed 0.9.1, 384 chiều; PostgreSQL 16.16, pgvector 0.8.7, cosine distance, exact search. Các version thư viện runtime liên quan: psycopg 3.3.8, pypdf 6.21.4, python-dotenv 1.2.4. Cài và chạy theo [starter README](../../starter/README.md): từ `starter/`, cài `python -m pip install -r requirements.txt`, tạo `.env`, chạy `docker compose -p day6_student_starter up -d --wait`, `python scripts/check_setup.py`, rồi `python -m src.run`. Chi phí API retrieval là $0 vì embedding chạy local; model initialization được ghi riêng trong run manifest.
- **Metadata:** `unit_id`, `source_id`, `document_id`, `source`, `chapter`, `section`, `position_start`, `position_end`, `char_count`, `text`. `chapter` là số La Mã từ heading và `section` là tiêu đề chương; hai trường này dùng lọc chính xác. `source` và `unit_id` hỗ trợ truy vết về Project Gutenberg và unit gốc. Metadata schema đầy đủ nằm trong manifest. 

## Cấu hình và benchmark
- **A:** fixed-size 500 ký tự, overlap mục tiêu 50; cửa sổ cuối được cân chỉnh để tránh fragment quá ngắn.
- **B:** paragraph-aware, nhóm paragraph theo thứ tự nguồn, tối đa 1,000 ký tự.
- **C:** cùng chunks/token counts/vectors như B; chỉ thêm exact-match filter `chapter`/`section` khi câu hỏi có filter.
- Benchmark có 20 câu tiếng Anh: 15 dev, 5 test; 12 câu có filter, trong đó 4 câu test. Mỗi câu gắn gold source/unit và evidence quote; pipeline xác minh quote tồn tại nguyên văn trong gold unit. Benchmark: [questions.jsonl](../../starter/benchmark/questions.jsonl). Trace lưu câu hỏi, top-3 chunk ID/text/score/metadata và kết quả evidence check.
- Tiêu chí chọn chiến lược **không được đăng ký trước khi xem test**; các giả thuyết trong experiment log được ghi rõ là retrospective. Năm test question/evidence pairs đã từng được đánh giá với câu hỏi tiếng Việt, vì vậy kết quả English hiện tại là exploratory, không phải untouched holdout. Không dùng test này để tuning hoặc tuyên bố chiến lược tốt nhất tổng quát. Không cần seed cho benchmark cố định và mẫu coherence chọn xác định theo thứ tự nguồn.

## Kết quả
| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Chunk count | Thời gian | Chi phí |
|---|---|---|---|---|---|---|
| A (Fixed-size) | Dev | 7/15 | 12/15 | 328 | Ingest 3.3213 s; retrieval 0.7544 s/20 câu (37.721 ms/query) | $0 |
| A (Fixed-size) | Test | 2/5 | 3/5 | 328 | Ingest 3.3213 s; retrieval 0.7544 s/20 câu (37.721 ms/query) | $0 |
| B (Theo đoạn văn) | Dev | 8/15 | 10/15 | 188 | Ingest 5.1941 s; retrieval 0.5663 s/20 câu (28.317 ms/query) | $0 |
| B (Theo đoạn văn)  | Test | 3/5 | 4/5 | 188 | Ingest 5.1941 s; retrieval 0.5663 s/20 câu (28.317 ms/query) | $0 |
| C (B + filter) | Dev | 8/15 | 11/15 | 188 | 0.1716 s lưu incremental; retrieval 0.5315 s/20 câu (26.575 ms/query) | $0 |
| C (B + filter)| Test | 3/5 | 4/5 | 188 | 0.1716 s lưu incremental; retrieval 0.5315 s/20 câu (26.575 ms/query) | $0 |

| Strategy | Coherence trung bình /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ (n/N hoặc N/A) | Grounding 0/1/2 (số câu) và trung bình /2 |
|---|---|---|---|---|
| A | 1.1/2 (10) | 15/15* | N/A — không áp dụng filter | 0/2/3; trung bình 1.6/2 |
| B | 1.7/2 (10) | 15/15* | N/A — không áp dụng filter | 0/1/4; trung bình 1.8/2 |
| C | 1.7/2 (10) | 15/15* | 4/4 câu test có filter | 0/1/4; trung bình 1.8/2 |

* Metadata: 15/15 kết quả mỗi strategy được tác giả rà soát, đối chiếu với corpus về source URL, page/unit ID, chapter và section. `page` là unit ID, không phải số trang in.
* Coherence phân bố nhãn 0/1/2: A 0/9/1, B 0/3/7, C 0/3/7. A chủ yếu nhận 1 vì phần đọc được vẫn có ý nghĩa nhưng bị cắt, thay vì chấm 0 chỉ vì mọi cửa sổ đều kết thúc đột ngột.

Tôi đã rà soát và xác nhận các nhãn coherence, grounding và metadata theo rubric, đối chiếu chunk/source và metadata với corpus. Kết quả tôi retrieval đã được xem; các nhãn 1 và mẫu coherence cần diễn giải thận trọng. Coherence template có 10 units chọn cách đều theo thứ tự nguồn × 3 chiến lược (30 dòng; mẫu được tạo xác định, không preregister trước khi xem metrics). Grounding template có 5 câu test × 3 chiến lược (15 dòng); metadata template có top-3 của 5 câu test × 3 chiến lược (45 dòng). Các file kết quả: [retrieval traces](../../starter/results/retrieval_traces.jsonl), [strategy metrics](../../starter/results/strategy_metrics.json), [run manifest](../../starter/results/run_manifest.json), [coherence audit template](../../starter/results/coherence_audit_template.jsonl), [grounding audit template](../../starter/results/grounding_audit_template.jsonl), [metadata audit template](../../starter/results/metadata_audit_template.jsonl). 

## Phân tích và kết luận
- **A–B:** A có Evidence Hit@3 cao hơn trên dev (12/15 so với 10/15), nhưng thấp hơn trên test exploratory (3/5 so với 4/5). A có 41,230 tokens, median 129 tokens/chunk; B có 36,611 tokens, median 212.5 tokens/chunk. A có nhiều chunks và overlap; B có ít chunks, nhưng kích thước biến thiên 98–999 ký tự.
- **B–C:** C tăng dev Evidence Hit@3 một câu (11/15 so với 10/15), test không đổi (4/5). C tuân thủ filter ở 4/4 câu test cần filter. C dùng lại chunks/vectors của B nên 0.1716 s chỉ là thời gian lưu incremental, không phải full-ingest time. Đây là lợi ích khi cần lọc metadata, không phải bằng chứng C tốt hơn mọi truy vấn.
- **Hai lỗi cụ thể:** `q17` (gold `alice-ch03-u02`, nguồn [Project Gutenberg ebook #11](https://www.gutenberg.org/ebooks/11)): cả A/B/C có gold unit trong top-3 nhưng không có chunk top-3 nào chứa trọn quote “We beg your acceptance of this elegant thimble”, nên Evidence Hit@3 thất bại do evidence không nằm trọn trong retrieved chunks. `q18` (gold `alice-ch04-u04`, cùng nguồn): A không đưa gold unit vào top-3; B/C có gold unit trong top-3 và chunk chứa đủ quote “The poor little Lizard, Bill”.
- **Chiến lược:** Không tuyên bố winner tổng quát. Nếu cần filter câu hỏi có chapter/section thì C tuân thủ 4/4 câu test có filter; nếu không, A/B có đánh đổi khác nhau giữa dev/test. Tác giả đã rà soát và xác nhận nhãn coherence/grounding/metadata theo rubric; kết quả không phải đánh giá mù.
- **Hạn chế:** một tác phẩm và 5 test câu; test đã từng dùng với câu hỏi tiếng Việt; người chấm đã xem output nên có thể thiên lệch; exact-quote metric có thể bỏ sót paraphrase hoặc evidence ghép nhiều chunks; `unit_id` không phải page number; thời gian là một lần chạy và phụ thuộc máy. Log hiện không có file lỗi riêng; lần chạy cuối thành công, lỗi batch embedding ban đầu đã được xử lý.
- **Nếu làm lại:** tạo test questions/evidence mới chưa từng xem; chốt tiêu chí lựa chọn trên dev trước; con người chấm độc lập xác nhận coherence, metadata và grounding labels; lặp lại timing trên cùng môi trường và lưu error log khi có lỗi.
