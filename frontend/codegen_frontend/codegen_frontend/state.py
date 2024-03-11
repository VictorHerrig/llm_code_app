import json
from uuid import uuid4

import requests
from requests.exceptions import HTTPError

import reflex as rx


def is_error(answer: str) -> bool:
    return answer is None or len(answer) < 5 or answer.startswith("ERROR")


class State(rx.State):
    query: str = None
    model_name: str = None
    lang_dict: dict[str, str] = {
        "Python": "python",
        "C++": "cpp",
        "C": "c",
        "Java": "java",
        "Scala": "scala",
        "SQL": "sql",
        "Javascript": "javascript",
        "bash": "bash",
    }
    model_dict: dict[str, str] = {
        'GPT-3.5': 'gpt-3.5-turbo-0125',
        'Hermes': 'teknium/OpenHermes-2.5-Mistral-7B'
    }
    inv_lang_dict: dict[str, str] = {v: k for k, v in lang_dict.items()}
    inv_model_dict: dict[str, str] = {v: k for k, v in model_dict.items()}
    backend_lang_dict: dict[str, str] = {
        "cpp": "c++"
    }

    language: str = None

    active_session: tuple[str, str, str, list[tuple[str, str, str, str, str, str, bool]]] = ()

    session_1_id = str(uuid4())
    session_2_id = str(uuid4())
    inactive_sessions: list[
        tuple[str, tuple[str, str, str, list[tuple[str, str, str, str, str, str, bool]]]]
    ] = []

    errors: list[str] = []
    new_session_id: str = ''
    _loading: bool = False
    _backend_generate_url: str = "http://backend:80/generate"
    _backend_adjust_url: str = "http://backend:80/adjust"

    @rx.var
    def active_session_id(self) -> str:
        if self.has_active_session:
            return self.active_session[0]
        return ''

    @rx.var
    def is_loading(self) -> bool:
        return self._loading

    @rx.var
    def session_language(self) -> str:
        if self.has_active_session:
            return self.active_session[1]
        return ''

    @rx.var
    def session_model(self) -> str:
        if self.has_active_session:
            return self.active_session[2]
        return ''

    @rx.var
    def session_history(self) -> list[tuple[str, str, str, str, str, str, bool]]:
        if self.has_active_session:
            return self.active_session[3]
        return []

    @rx.var
    def has_active_session(self) -> bool:
        return self.active_session is not None and len(self.active_session) > 0

    @rx.var
    def has_inactive_sessions(self) -> bool:
        return self.inactive_sessions is not None and len(self.inactive_sessions) > 0 and len(self.inactive_sessions[0]) == 2 and len(self.inactive_sessions[0][1]) > 0

    @rx.var
    def inactive_session_history(
        self,
    ) -> list[tuple[str, str, str, list[tuple[str, str, str, str, str, str, bool]]]]:
        if self.has_inactive_sessions:
            return [s[1] for s in self.inactive_sessions]
        else:
            return []

    def api_call(self):
        # Fetch language and model name in backend-readable format
        language = self.active_session[1]
        model_name = self.active_session[2]
        if language in self.backend_lang_dict:
            language = self.backend_lang_dict[language]

        if len(self.active_session[3]) == 0:
            # Generation
            payload = dict(
                model_name=model_name,
                language=language,
                original_prompt=self.query
            )
            response = requests.post(self._backend_generate_url, json=payload)
            if response.status_code == 200:
                return json.loads(response.content.decode('utf-8'))['code_out']
            else:
                raise HTTPError(f"{response.status_code}: {response.content}")
        else:
            # Adjust previous code
            first_query = self.active_session[3][-1][4]
            last_answer = self.active_session[3][0][5]
            payload = dict(
                model_name=model_name,
                language=language,
                original_prompt=first_query,
                reference_code=last_answer,
                feedback_prompt=self.query
            )
            response = requests.post(self._backend_adjust_url, json=payload)
            if response.status_code == 200:
                return json.loads(response.content.decode('utf-8'))['code_out']
            else:
                raise HTTPError(f"{response.status_code}: {response.content}")

    def submit(self):
        self.errors = []
        if self.query is None or len(self.query) < 5:
            self.errors.append("Please input a prompt of at least 5 character")
        if self.query is not None and len(self.query) > 200:
            self.errors.append("Please limit input a prompt length to 200 character")
        if len(self.errors) == 0:
            try:
                answer = self.api_call()
                if answer is None:
                    self.errors.append("Backend error")
                turn = (
                    self.active_session[0],  # session_id
                    str(uuid4()),            # turn_id
                    self.active_session[1],  # language_id
                    self.active_session[2],  # model_id
                    self.query,              # query
                    answer,                  # answer
                    is_error(answer),        # answer_is_error
                )
                self.active_session[3].insert(0, turn)
            except HTTPError as e:
                self.errors.append(f"Backend error: {str(e)}")

    def new_session(self):
        self.errors = []
        if self.language is None or len(self.language) == 0:
            self.errors.append("Please select a language")
        if self.model_name is None or len(self.model_name) == 0:
            self.errors.append("Please select a model")
        if len(self.errors) == 0:
            if self.has_active_session and len(self.active_session[3]) > 0:
                self.inactive_sessions.append((self.active_session[0], self.active_session))

            session_id = str(uuid4())
            self.active_session = (session_id, self.lang_dict[self.language], self.model_dict[self.model_name], [])

    def change_session(self):
        if self.new_session_id is not None and any(
            [i == self.new_session_id for i, _ in self.inactive_sessions]
        ):
            idx = [
                j
                for j, (i, _) in enumerate(self.inactive_sessions)
                if i == self.new_session_id
            ][0]
            new_activate_session = self.inactive_sessions[idx][1]
            self.inactive_sessions.pop(idx)

            if self.has_active_session and len(self.active_session[1]) > 0:
                inactive_session_id = self.active_session[0]
                self.inactive_sessions.append(
                    (inactive_session_id, self.active_session)
                )

            self.active_session = new_activate_session
            self.language = self.active_session[0]

            self.new_session_id = None
