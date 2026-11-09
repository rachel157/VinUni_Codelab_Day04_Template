"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
  - Hỗ trợ cả 2 chế độ:
      1. Mock Simulator (mặc định, không cần API key)
      2. Live Gemini API (khi có GEMINI_API_KEY)
"""

import sys
import os
import json
import re
from typing import Dict, Any, List

# Cấu hình stdout UTF-8 cho môi trường Windows Console
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """Bạn là VinAssistant — Trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA (Định danh & Tính cách)
- Tên: VinAssistant.
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng của hệ sinh thái Vingroup (tập trung vào VinFast và Vinpearl).
- Phong cách giao tiếp: Chuyên nghiệp, thân thiện, lịch thiệp, chu đáo và sử dụng tiếng Việt chuẩn mực.

## 2. AVAILABLE TOOLS (Danh mục công cụ)
Bạn được cung cấp 2 công cụ chính thức để tra cứu dữ liệu và tương tác hệ thống:
1. `search_product_catalog(category: str, max_price: int)`:
   - Tra cứu danh sách sản phẩm hoặc dịch vụ theo phân loại ('xe_dien' hoặc 'du_lich') và mức giá trần tính bằng VNĐ.
2. `submit_support_ticket(customer_name: str, issue_description: str, priority: str)`:
   - Ghi nhận yêu cầu hỗ trợ, khiếu nại chất lượng dịch vụ hoặc báo lỗi kỹ thuật của khách hàng vào hệ thống với các mức độ ưu tiên ('low', 'medium', 'high').

## 3. CORE RULES (Nguyên tắc vận hành bắt buộc)
1. Tuyệt đối KHÔNG hallucinate (tự bịa đặt) giá bán, thông số kỹ thuật hoặc mã phiếu hỗ trợ.
2. BẮT BUỘC gọi `search_product_catalog` khi khách hỏi thông tin giá cả, sản phẩm, gói dịch vụ nghỉ dưỡng.
3. BẮT BUỘC gọi `submit_support_ticket` khi khách thông báo sự cố, phàn nàn hoặc cần tạo yêu cầu xử lý.
4. Nếu khách hàng có nhiều ý định cùng lúc (vừa muốn tìm sản phẩm, vừa gửi khiếu nại), phải kích hoạt cả hai công cụ tương ứng.
5. Nếu công cụ tìm kiếm trả về danh sách trống, hãy thông báo rõ ràng và đề xuất phương án hỗ trợ khác, không được tự bịa ra sản phẩm.

## 4. OPERATIONAL BOUNDARIES (Phạm vi nghiệp vụ)
- Chỉ giải quyết các câu hỏi liên quan đến sản phẩm, dịch vụ và chính sách của Vingroup.
- Lịch sự từ chối trả lời các chủ đề ngoài phạm vi (chính trị, so sánh tiêu cực đối thủ, công nghệ ngoài hệ sinh thái).
- Đối với các câu hỏi chính sách đã công bố cố định (FAQ) như bảo hành pin VinFast 10 năm, trả lời trực tiếp rõ ràng mà không cần gọi tool.

## 5. OUTPUT CONTRACT (Quy ước định dạng phản hồi)
Tuân theo định dạng suy luận ReAct để đảm bảo tính minh bạch:
- Thought: Phân tích yêu cầu khách hàng, xác định intent và tham số cho tool.
- Action: Tên công cụ cần gọi kèm tham số (hoặc None nếu là câu hỏi FAQ thông thường).
- Observation: Dữ liệu thực tế nhận về từ hệ thống sau khi tool thực thi.
- Final Answer: Câu trả lời hoàn chỉnh, ngắn gọn, lịch sự gửi đến khách hàng dựa trên dữ liệu thật.
"""


# ═══════════════════════════════════════════════════════════════════════════
# HELPER: Dynamic Model Discovery with Cache & Rest Transport
# ═══════════════════════════════════════════════════════════════════════════

_CACHED_MODEL_NAME = None

def get_active_gemini_model(api_key: str) -> str:
    """
    Tự động dò tìm mô hình Gemini khả dụng và mới nhất cho API key hiện tại.
    Có bộ nhớ cache để không phải gọi API list_models nhiều lần làm chậm hoặc treo.
    """
    global _CACHED_MODEL_NAME
    if _CACHED_MODEL_NAME:
        return _CACHED_MODEL_NAME

    preferred_models = [
        "gemini-3.8-flash",
        "gemini-3.8-flash-latest",
        "gemini-3.8-pro",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-pro"
    ]
    try:
        import google.generativeai as genai
        # Sử dụng transport="rest" trên Windows để tránh lỗi treo kết nối của gRPC
        genai.configure(api_key=api_key, transport="rest")
        available_models = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in getattr(m, "supported_generation_methods", [])
        ]
        for pref in preferred_models:
            if pref in available_models:
                _CACHED_MODEL_NAME = pref
                return pref
        for m in available_models:
            if "3.8" in m or "3." in m:
                _CACHED_MODEL_NAME = m
                return m
        for m in available_models:
            if "flash" in m:
                _CACHED_MODEL_NAME = m
                return m
        if available_models:
            _CACHED_MODEL_NAME = available_models[0]
            return _CACHED_MODEL_NAME
    except Exception:
        pass

    _CACHED_MODEL_NAME = "gemini-3.8-flash"
    return _CACHED_MODEL_NAME


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Nếu có GEMINI_API_KEY: gọi Google Gemini 1 lượt (không dùng tool để thấy hallucination).
        Nếu không có: trả về kết quả mock baseline.
        """
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key, transport="rest")
                model_name = get_active_gemini_model(self.api_key)
                print(f"[Chatbot Baseline] Đang gọi Gemini ({model_name})...")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    f"Bạn là trợ lý ảo. Hãy trả lời câu hỏi sau không dùng công cụ: {user_input}",
                    request_options={"timeout": 45}
                )
                return {
                    "answer": response.text,
                    "tool_calls": [],
                    "status": "success",
                    "mode": f"live_gemini_api ({model_name})"
                }
            except Exception as e:
                # Nếu Google gợi ý model mới trong thông báo lỗi, tự động thử lại
                match_suggest = re.search(r"use models/([a-zA-Z0-9._-]+)", str(e))
                if match_suggest:
                    try:
                        suggested_model = match_suggest.group(1)
                        model = genai.GenerativeModel(suggested_model)
                        response = model.generate_content(
                            f"Bạn là trợ lý ảo. Hãy trả lời câu hỏi sau không dùng công cụ: {user_input}",
                            request_options={"timeout": 15}
                        )
                        return {
                            "answer": response.text,
                            "tool_calls": [],
                            "status": "success",
                            "mode": f"live_gemini_api ({suggested_model})"
                        }
                    except Exception:
                        pass

        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering, Tool Calling & Safeguards."""

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []

    def _extract_name(self, text: str) -> str:
        """Trích xuất tên khách hàng từ câu hỏi."""
        m = re.search(
            r"(?:tôi tên là|tôi tên|tên tôi là)\s*([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|$|\.|\b(?:xe|phòng|và|đây)\b)",
            text,
            re.IGNORECASE
        )
        return m.group(1).strip() if m else "Khách hàng"

    def _extract_issue(self, text: str) -> str:
        """Trích xuất mô tả vấn đề/sự cố cần hỗ trợ."""
        m = re.search(
            r"((?:xe|phòng|sản phẩm|dịch vụ)?\s*[^,.]*?(?:bị lỗi|lỗi|ẩm mốc|sự cố|hỏng)[^,.]*)",
            text,
            re.IGNORECASE
        )
        return m.group(1).strip() if m else "Yêu cầu hỗ trợ kỹ thuật"

    def _extract_priority(self, text: str) -> str:
        """Trích xuất mức độ ưu tiên của yêu cầu."""
        if re.search(r"\b(gấp|nghiêm trọng|khẩn cấp|high)\b", text, re.IGNORECASE):
            return "high"
        if re.search(r"\b(thấp|không gấp|low)\b", text, re.IGNORECASE):
            return "low"
        return "medium"

    def _extract_price(self, text: str) -> int:
        """Trích xuất mức giá tối đa (VNĐ)."""
        m = re.search(
            r"(?:dưới|<|khoảng|tối đa)\s*(\d+(?:[.,]\d+)?)\s*(triệu|tr|tỷ|ty)?",
            text,
            re.IGNORECASE
        )
        if m:
            val = float(m.group(1).replace(",", "."))
            unit = m.group(2)
            if unit and unit.lower() in ["tỷ", "ty"]:
                return int(val * 1_000_000_000)
            return int(val * 1_000_000)
        return 999999999999

    def _extract_category(self, text: str) -> str:
        """Phân loại danh mục sản phẩm (xe_dien hoặc du_lich)."""
        if re.search(r"\b(resort|du lịch|du_lich|khách sạn|nghỉ dưỡng|vinpearl|phòng)\b", text, re.IGNORECASE):
            return "du_lich"
        return "xe_dien"

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Điểm vào chính:
        Nếu có GEMINI_API_KEY -> gọi Live Gemini API với Tool Calling.
        Nếu không -> chạy Mock Simulator.
        """
        if self.api_key:
            return self._run_live_gemini(user_input)
        return self._run_mock(user_input)

    def _run_live_gemini(self, user_input: str) -> Dict[str, Any]:
        """Gọi Google Gemini Live API với Function Calling tự động."""
        self.trace = []
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key, transport="rest")

            # Tự động chọn mô hình khả dụng và tương thích nhất
            model_name = get_active_gemini_model(self.api_key)
            print(f"[ToolCallingAgent] Đang gửi yêu cầu đến Gemini '{model_name}' (timeout: 45s)...")

            # Khởi tạo mô hình Gemini kèm Tool Definitions và System Prompt
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
                tools=[search_product_catalog, submit_support_ticket]
            )

            # Khởi tạo chat có kích hoạt tự động thực thi Tool trong Python
            chat = model.start_chat(enable_automatic_function_calling=True)
            response = chat.send_message(user_input, request_options={"timeout": 45})

            # Thu thập vết gọi tool từ lịch sử hội thoại
            step_count = 0
            for message in chat.history:
                for part in message.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        step_count += 1
                        call = part.function_call
                        self.trace.append({
                            "iteration": step_count,
                            "role": message.role,
                            "action": call.name,
                            "action_input": dict(call.args)
                        })
                    elif hasattr(part, "function_response") and part.function_response:
                        resp = part.function_response
                        self.trace.append({
                            "role": message.role,
                            "observation": dict(resp.response) if hasattr(resp.response, "items") else str(resp.response)
                        })

            return {
                "answer": response.text,
                "trace": self.trace,
                "iterations": max(step_count, 1),
                "status": "completed",
                "mode": f"live_gemini_api ({model_name})"
            }
        except Exception as e:
            err_msg = str(e)
            # Tự động trích xuất model được Google gợi ý (Self-healing)
            match_suggest = re.search(r"use models/([a-zA-Z0-9._-]+)", err_msg)
            if match_suggest:
                suggested_model = match_suggest.group(1)
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=self.api_key, transport="rest")
                    print(f"[ToolCallingAgent] Đang thử lại với mô hình Google gợi ý: '{suggested_model}'...")
                    model = genai.GenerativeModel(
                        model_name=suggested_model,
                        system_instruction=SYSTEM_PROMPT,
                        tools=[search_product_catalog, submit_support_ticket]
                    )
                    chat = model.start_chat(enable_automatic_function_calling=True)
                    response = chat.send_message(user_input, request_options={"timeout": 45})
                    step_count = sum(1 for m in chat.history for p in m.parts if hasattr(p, "function_call") and p.function_call)
                    return {
                        "answer": response.text,
                        "trace": self.trace,
                        "iterations": max(step_count, 1),
                        "status": "completed",
                        "mode": f"live_gemini_api ({suggested_model})"
                    }
                except Exception as retry_err:
                    print(f"[Cảnh báo] Lỗi khi gọi Gemini Live API ({suggested_model}): {retry_err}. Đang chuyển về Mock Simulator...")
            else:
                print(f"[Cảnh báo] Lỗi kết nối/timeout Gemini Live API ({e}). Đang chuyển về Mock Simulator...")
            return self._run_mock(user_input)

    def _run_mock(self, user_input: str) -> Dict[str, Any]:
        """Chạy Agent Loop ở chế độ Mock Simulator (dành cho kiểm thử offline & autograder)."""
        self.trace = []

        # -------------------------------------------------------------------
        # TODO 3: Phân tích intent từ user_input (Độc lập, không dùng if-elif)
        # -------------------------------------------------------------------
        is_faq = bool(
            re.search(r"(bảo hành pin|chính sách bảo hành|kéo dài bao lâu)", user_input, re.IGNORECASE)
            and not re.search(r"(giá dưới|mua|bị lỗi|tôi tên)", user_input, re.IGNORECASE)
        )
        needs_ticket = bool(
            re.search(r"(bị lỗi|sự cố|phản hồi|khiếu nại|tôi tên|tên tôi là|xử lý gấp|hỗ trợ kỹ thuật)", user_input, re.IGNORECASE)
        )
        needs_catalog = bool(
            re.search(r"(xem|tìm|mua|danh sách|có xe nào|có resort nào|giá dưới|giá|catalog)", user_input, re.IGNORECASE)
            and not is_faq
        )

        # -------------------------------------------------------------------
        # TODO 4 (Milestone 4.1): Max Iterations Guard
        # -------------------------------------------------------------------
        iteration = 1
        if iteration > self.max_iterations:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": iteration,
                "status": "max_iterations_reached"
            }

        # Kịch bản 1: Câu hỏi FAQ thông thường (không cần gọi tool)
        if is_faq:
            thought = "Khách hàng hỏi về chính sách bảo hành pin. Đây là câu hỏi FAQ đã có quy định chuẩn, trả lời trực tiếp mà không cần gọi tool."
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc không giới hạn số km (tùy theo từng dòng xe), mang lại sự an tâm tối đa cho quý khách."
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": None,
                "observation": "FAQ Answered directly.",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Kịch bản 2: Tra cứu danh mục sản phẩm (Single tool: search_product_catalog)
        if needs_catalog and not needs_ticket:
            category = self._extract_category(user_input)
            max_price = self._extract_price(user_input)
            thought = f"Khách hàng cần tìm sản phẩm danh mục '{category}' với giá tối đa {max_price:,} VNĐ. Gọi tool search_product_catalog."

            # Action
            results = search_product_catalog(category=category, max_price=max_price)

            # Observation
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": "search_product_catalog",
                "action_input": {"category": category, "max_price": max_price},
                "observation": results
            })

            # Milestone 4.2: Empty Results Handling (Xử lý khi kết quả rỗng)
            if not results or (len(results) == 1 and "error" in results[0]):
                answer = f"Rất tiếc, chúng tôi không tìm thấy sản phẩm nào phù hợp với yêu cầu giá dưới {max_price:,} VNĐ của quý khách."
            else:
                product_lines = []
                for p in results:
                    price_vnd = p.get("price_vnd", 0)
                    product_lines.append(f"- {p.get('name')}: {price_vnd:,} VNĐ ({p.get('description', '')})")
                answer = f"Dưới đây là các sản phẩm phù hợp với yêu cầu của quý khách:\n" + "\n".join(product_lines)

            self.trace.append({
                "iteration": iteration,
                "thought": "Đã nhận được kết quả tra cứu, tổng hợp câu trả lời gửi đến khách hàng.",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Kịch bản 3: Tiếp nhận khiếu nại / sự cố (Single tool: submit_support_ticket)
        if needs_ticket and not needs_catalog:
            customer_name = self._extract_name(user_input)
            issue_description = self._extract_issue(user_input)
            priority = self._extract_priority(user_input)

            thought = f"Khách hàng {customer_name} phản ánh sự cố '{issue_description}' mức ưu tiên '{priority}'. Gọi tool submit_support_ticket."

            # Action
            ticket_res = submit_support_ticket(
                customer_name=customer_name,
                issue_description=issue_description,
                priority=priority
            )

            # Observation
            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": "submit_support_ticket",
                "action_input": {
                    "customer_name": customer_name,
                    "issue_description": issue_description,
                    "priority": priority
                },
                "observation": ticket_res
            })

            ticket_id = ticket_res.get("ticket_id", "TK-UNKNOWN")
            answer = (
                f"Kính chào quý khách {customer_name}, yêu cầu hỗ trợ của quý khách về vấn đề '{issue_description}' "
                f"đã được tiếp nhận thành công vào hệ thống với mã số {ticket_id} (mức độ ưu tiên: {priority}). "
                f"Đội ngũ kỹ thuật viên VinFast sẽ sớm liên hệ để hỗ trợ và xử lý sự cố cho quý khách."
            )

            self.trace.append({
                "iteration": iteration,
                "thought": "Đã tạo ticket thành công, gửi phản hồi xác nhận cho khách hàng.",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Kịch bản 4: Đa ý định (Multi-intent: vừa tra cứu vừa tạo ticket)
        if needs_catalog and needs_ticket:
            # Iteration 1: Gọi search_product_catalog
            category = self._extract_category(user_input)
            max_price = self._extract_price(user_input)
            thought1 = f"Bước 1: Tra cứu danh mục '{category}' giá dưới {max_price:,} VNĐ."
            results = search_product_catalog(category=category, max_price=max_price)
            self.trace.append({
                "iteration": 1,
                "thought": thought1,
                "action": "search_product_catalog",
                "action_input": {"category": category, "max_price": max_price},
                "observation": results
            })

            # Guard trước Iteration 2
            iteration = 2
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            # Iteration 2: Gọi submit_support_ticket
            customer_name = self._extract_name(user_input)
            issue_description = self._extract_issue(user_input)
            priority = self._extract_priority(user_input)
            thought2 = f"Bước 2: Tạo phiếu hỗ trợ cho khách hàng {customer_name}."
            ticket_res = submit_support_ticket(
                customer_name=customer_name,
                issue_description=issue_description,
                priority=priority
            )
            self.trace.append({
                "iteration": 2,
                "thought": thought2,
                "action": "submit_support_ticket",
                "action_input": {
                    "customer_name": customer_name,
                    "issue_description": issue_description,
                    "priority": priority
                },
                "observation": ticket_res
            })

            # Guard trước Iteration 3
            iteration = 3
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa.",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            # Iteration 3: Tổng hợp phản hồi
            ticket_id = ticket_res.get("ticket_id", "TK-UNKNOWN")
            catalog_summary = []
            if not results or (len(results) == 1 and "error" in results[0]):
                catalog_text = f"Không tìm thấy dịch vụ nào phù hợp dưới {max_price:,} VNĐ."
            else:
                for p in results:
                    catalog_summary.append(f"- {p.get('name')}: {p.get('price_vnd', 0):,} VNĐ")
                catalog_text = "\n".join(catalog_summary)

            answer = (
                f"Kính chào quý khách {customer_name}!\n\n"
                f"1. Về danh sách sản phẩm/dịch vụ phù hợp (dưới {max_price:,} VNĐ):\n"
                f"{catalog_text}\n\n"
                f"2. Về phản ánh của quý khách:\n"
                f"Hệ thống đã tiếp nhận phản ánh về vấn đề '{issue_description}' với mã số {ticket_id} (mức ưu tiên: {priority}). Bộ phận hỗ trợ sẽ xử lý trong thời gian sớm nhất."
            )
            self.trace.append({
                "iteration": iteration,
                "thought": "Tổng hợp kết quả từ cả hai công cụ thành câu trả lời hoàn chỉnh.",
                "final_answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Fallback chung nếu không thuộc intent nào
        return {
            "answer": "Rất tiếc, tôi chưa hiểu rõ yêu cầu của quý khách. Quý khách vui lòng cung cấp thêm thông tin.",
            "trace": self.trace,
            "iterations": 1,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nghiệm nhiều tình huống thực tế
# ═══════════════════════════════════════════════════════════════════════════

def main():
    agent = ToolCallingAgent(max_iterations=5)
    baseline = ChatbotBaseline()

    mode_str = "LIVE GEMINI API" if agent.api_key else "MOCK SIMULATOR (Offline)"
    print(f"=== CHẾ ĐỘ ĐANG CHẠY: {mode_str} ===")

    test_queries = [
        ("1. Tìm xe điện dưới 600 triệu (Single Tool Catalog)",
         "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."),
        ("2. Báo lỗi xe VF 8 (Single Tool Ticket)",
         "Tôi tên Lê Minh Khoa, xe VF 8 của tôi bị lỗi hệ thống ADAS. Đây là vấn đề nghiêm trọng, cần xử lý gấp."),
        ("3. Đa ý định (Parallel/Multi-intent: Resort + Báo phòng ẩm mốc)",
         "Cho tôi xem resort Vinpearl giá dưới 6 triệu. Và tôi cũng muốn ghi nhận phản hồi: tên tôi là Phạm Thị Dung, phòng tôi đặt ở Vinpearl Nha Trang bị ẩm mốc, mức độ trung bình."),
        ("4. Hỏi chính sách bảo hành (FAQ không cần tool)",
         "Chính sách bảo hành pin xe điện VinFast kéo dài bao lâu?"),
        ("5. Tìm xe giá dưới 200 triệu (Empty Results Fallback)",
         "Có xe điện nào giá dưới 200 triệu không?")
    ]

    print("\n======================================================================")
    print("DEMO 1: CHATBOT BASELINE (Không dùng tool - Dễ hallucination)")
    print("======================================================================")
    demo_q = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."
    res_baseline = baseline.query(demo_q)
    print("User Query :", demo_q)
    print("Answer     :", res_baseline["answer"])
    print("Tool Calls :", res_baseline["tool_calls"])
    print("Mode       :", res_baseline.get("mode"))

    print("\n======================================================================")
    print("DEMO 2: TOOL CALLING AGENT (5 Tình huống thực tế)")
    print("======================================================================")
    for title, q in test_queries:
        print(f"\n--- {title} ---")
        print("User:", q)
        res = agent.run(q)
        print("Agent Answer:\n" + res["answer"])
        print(f"[Trạng thái: {res['status']} | Số bước: {res['iterations']} | Mode: {res.get('mode', 'mock')}]")

if __name__ == "__main__":
    main()
