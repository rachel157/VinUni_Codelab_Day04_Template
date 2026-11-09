"""
Lab #4 Test Runner — Kiểm thử tự động 8 Unit Tests
Không yêu cầu cài đặt thêm thư viện bên ngoài (dùng chuẩn Python có sẵn).

Chạy lệnh:
    python run_tests.py
"""

import sys
import os

# Cấu hình UTF-8 cho console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thêm đường dẫn starter-code
STARTER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "starter-code"))
if STARTER_DIR not in sys.path:
    sys.path.insert(0, STARTER_DIR)

from template import ChatbotBaseline, ToolCallingAgent
from tools import search_product_catalog, submit_support_ticket

def run_all_tests():
    tests = []

    # Test 1
    def t1():
        results = search_product_catalog(category="xe_dien", max_price=600000000)
        assert isinstance(results, list), "Kết quả phải là danh sách"
        assert len(results) == 2, f"Kỳ vọng 2 xe, nhận được {len(results)}"
        names = [p["name"] for p in results]
        assert "VinFast VF 3" in names and "VinFast VF 5 Plus" in names
    tests.append(("Test 1: search_product_catalog — xe điện giá dưới 600 triệu", t1))

    # Test 2
    def t2():
        results = search_product_catalog(category="du_lich", max_price=6000000)
        assert isinstance(results, list), "Kết quả phải là danh sách"
        assert len(results) == 2, f"Kỳ vọng 2 gói nghỉ dưỡng, nhận được {len(results)}"
        names = [p["name"] for p in results]
        assert "Vinpearl Resort & Spa Nha Trang" in names
        assert "Vinpearl Discovery Phú Quốc" in names
    tests.append(("Test 2: search_product_catalog — du lịch giá dưới 6 triệu", t2))

    # Test 3
    def t3():
        result = submit_support_ticket(
            customer_name="Test User",
            issue_description="Test issue for autograder",
            priority="high"
        )
        assert isinstance(result, dict)
        assert "ticket_id" in result and result["ticket_id"].startswith("TK-")
        assert result["customer_name"] == "Test User"
        assert result["priority"] == "high"
        assert result["status"] == "open"
    tests.append(("Test 3: submit_support_ticket — tạo ticket thành công", t3))

    # Test 4
    def t4():
        chatbot = ChatbotBaseline()
        res = chatbot.query("Tôi muốn xem xe VinFast")
        assert res["status"] == "success"
        assert len(res["tool_calls"]) == 0
    tests.append(("Test 4: ChatbotBaseline — không gọi tool", t4))

    # Test 5
    def t5():
        agent = ToolCallingAgent(max_iterations=5)
        res = agent.run("Tôi muốn xem xe điện VinFast giá dưới 600 triệu.")
        assert res["status"] == "completed"
        assert res["iterations"] == 1
        assert "VF 3" in res["answer"] or "VF 5" in res["answer"]
    tests.append(("Test 5: ToolCallingAgent — single tool (catalog search)", t5))

    # Test 6
    def t6():
        agent = ToolCallingAgent(max_iterations=5)
        res = agent.run("Tôi tên Lê Minh Khoa, xe VF 8 của tôi bị lỗi hệ thống ADAS. Đây là vấn đề nghiêm trọng, cần xử lý gấp.")
        assert res["status"] == "completed"
        assert res["iterations"] == 1
        assert "TK-" in res["answer"]
        assert "Lê Minh Khoa" in res["answer"]
    tests.append(("Test 6: ToolCallingAgent — single tool (ticket submit)", t6))

    # Test 7
    def t7():
        agent = ToolCallingAgent(max_iterations=5)
        res = agent.run("Chính sách bảo hành pin xe điện VinFast kéo dài bao lâu?")
        assert res["status"] == "completed"
        assert res["iterations"] == 1
        assert "bảo hành" in res["answer"].lower() or "10 năm" in res["answer"]
    tests.append(("Test 7: ToolCallingAgent — FAQ (no tool call)", t7))

    # Test 8
    def t8():
        agent = ToolCallingAgent(max_iterations=5)
        res = agent.run("Có xe điện nào giá dưới 200 triệu không?")
        assert res["status"] == "completed"
        assert "không tìm thấy" in res["answer"].lower() or "rất tiếc" in res["answer"].lower()
    tests.append(("Test 8: ToolCallingAgent — no results fallback", t8))

    print("=" * 70)
    print("BẮT ĐẦU CHẠY 8 BÀI TEST KIỂM THỬ TỰ ĐỘNG (LAB #4)")
    print("=" * 70)

    passed_count = 0
    for idx, (name, test_fn) in enumerate(tests, 1):
        try:
            test_fn()
            print(f"[{idx}/8]  PASSED  : {name}")
            passed_count += 1
        except Exception as e:
            print(f"[{idx}/8]  FAILED  : {name}")
            print(f"          Lỗi: {e}")

    print("=" * 70)
    if passed_count == len(tests):
        print(f"KẾT QUẢ: XUẤT SẮC! ĐẠT {passed_count}/{len(tests)} TESTS PASSED!")
    else:
        print(f"KẾT QUẢ: ĐẠT {passed_count}/{len(tests)} TESTS.")
    print("=" * 70)

if __name__ == "__main__":
    run_all_tests()
