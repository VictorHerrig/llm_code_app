import reflex as rx

from codegen_frontend.state import State
from rxconfig import config

filename = f"{config.app_name}/{config.app_name}.py"


# ------------------------------- #
# ------ Session selector ------- #
# ------------------------------- #


def language_select() -> rx.Component:
    return rx.chakra.flex(
        rx.chakra.select(
            [
                "Python",
                "C++",
                "SQL",
                "Java",
                "Scala",
                "bash",
                "Javascript",
                "Typescript",
            ],
            placeholder="Select language",
            on_change=State.set_language,
        )
    )


def model_select() -> rx.Component:
    return rx.chakra.flex(
        rx.chakra.select(
            ["GPT-3.5", "Hermes"],
            placeholder="Select model",
            on_change=State.set_model_name,
        )
    )


def new_session_button() -> rx.Component:
    return rx.chakra.flex(
        rx.chakra.button(
            "Start new session", on_click=State.new_session(), color_scheme="gray"
        )
    )


def session_bar() -> rx.Component:
    return rx.chakra.hstack(
        language_select(), model_select(), new_session_button(), margin_y="2em"
    )


# ------------------------------- #
# ------- Generation UI --------- #
# ------------------------------- #


def language_model_label() -> rx.Component:
    return rx.chakra.box(
        f"Current Session: Language: {State.inv_lang_dict[State.session_language]}, "
        f"Model: {State.inv_model_dict[State.session_model]}"
    )


def action_label() -> rx.Component:
    return rx.chakra.box(f"{State.call_type}")


def query_input() -> rx.Component:
    return rx.chakra.text_area(
        placeholder="What would you like to do?",
        on_change=State.set_query,
        wrap="wrap",
        max_length=200,
        style={"height": 100, "width": 400},
    )


def query_button() -> rx.Component:
    return rx.chakra.button(
        "Send",
        on_click=State.submit(),
        color_scheme="gray",
        is_loading=State.is_loading,
    )


def action_bar() -> rx.Component:
    return rx.cond(
        State.has_active_session,
        rx.chakra.vstack(
            language_model_label(),
            action_label(),
            rx.chakra.hstack(query_input(), query_button(), margin_y="2em"),
        ),
    )


# ------------------------------- #
# ------------ Errors ----------- #
# ------------------------------- #


def error_box(error_val: str) -> rx.Component:
    return rx.chakra.box(
        rx.chakra.text(error_val, high_contrast=True),
        background_color="rgb(255, 70, 70)",
        margin_y="1em",
    )


def error_boxes() -> rx.Component:
    return rx.foreach(State.errors, lambda error: error_box(error))


# ------------------------------- #
# --- Active session history ---- #
# ------------------------------- #


def code_or_error(turn: tuple[str, str, str, str, str, str, bool]) -> rx.Component:
    is_error = turn[6]
    answer = turn[5]
    language = State.inv_lang_dict[turn[2]]
    return rx.cond(
        is_error,
        rx.chakra.box(
            rx.chakra.text(answer, color="rgb(255, 70, 70)"),
            text_align="left",
            margin_x="1em",
        ),
        rx.code_block(
            answer,
            text_align="left",
            show_line_numbers=True,
            language=language,
            margin_x="1em"
        ),
    )


def active_turn(turn: tuple[str, str, str, str, str, str, bool]) -> rx.Component:
    query = turn[4]
    return rx.chakra.box(
        rx.chakra.box(rx.text(query, color="black"), text_align="left", margin="1em"),
        code_or_error(turn),
        margin_y="1em",
        border_color="gray",
        bg="LightGray",
    )


def active_session() -> rx.Component:
    return rx.cond(
        State.has_active_session,
        rx.chakra.flex(
            rx.chakra.vstack(
                rx.foreach(State.session_history, lambda turn: active_turn(turn)),
                style={"background_color": "white"},
            )
        ),
    )


# ------------------------------- #
# -- Inactive session history --- #
# ------------------------------- #


def inactive_turn(turn: tuple[str, str, str, str, str, str, bool]) -> rx.Component:
    query = turn[4]
    return rx.chakra.box(
        rx.chakra.box(rx.text(query, color="black"), text_align="left", margin="1em"),
        code_or_error(turn),
        margin_y="1em",
        margin_x="4em",
        border_color="gray",
        bg="LightGray",
    )


def inactive_session(
    session: tuple[str, str, str, list[tuple[str, str, str, str, str, str, bool]]]
) -> rx.Component:
    history = session[3]
    return rx.chakra.hstack(
        rx.chakra.vstack(
            rx.foreach(history, lambda turn: inactive_turn(turn)),
            style={"background_color": "white"},
        ),
        rx.vstack(
            rx.chakra.button(
                rx.chakra.text("Switch to this Session", style={"color": "black"}),
                color_scheme="gray",
                on_click=State.change_session(),
            ),
            rx.spacer(align="stretch"),
        ),
        style={"background_color": "white"},
    )


def inactive_session_accordion(
    session: tuple[str, str, str, list[tuple[str, str, str, str, str, str, bool]]]
) -> rx.Component:
    session_id = session[0]
    language_id = session[1]
    model_id = session[2]
    history = session[3]
    code_lang = State.inv_lang_dict[language_id]
    model_name = State.inv_model_dict[model_id]
    orig_code_prompt = history[-1][4]

    return rx.accordion.item(
        header=rx.chakra.text(
            f"{code_lang} - {model_name}: {orig_code_prompt}", style={"color": "black"}
        ),
        content=rx.chakra.flex(inactive_session(session), direction="column"),
        value=session_id,
        style={"background_color": "white"},
    )


def inactive_sessions() -> rx.Component:
    return rx.chakra.flex(
        rx.cond(
            State.has_inactive_sessions,
            rx.chakra.vstack(
                rx.chakra.box("Session History"),
                rx.accordion.root(
                    rx.foreach(
                        State.inactive_session_history,
                        lambda session: inactive_session_accordion(session),
                    ),
                    collapsible=True,
                    width="50em",
                    type="single",
                    style={"background_color": "white"},
                    color_scheme="gray",
                    value=State.new_session_id,
                    on_value_change=State.set_new_session_id,
                ),
                rx.chakra.divider(margin_y="4em"),
            ),
        ),
        direction="row",
        spacing="2",
    )


# ------------------------------- #
# ------------ Page ------------- #
# ------------------------------- #


def index() -> rx.chakra.Component:
    return rx.chakra.container(
        session_bar(),
        rx.chakra.divider(margin_y="2em"),
        inactive_sessions(),
        action_bar(),
        error_boxes(),
        active_session(),
    )


app = rx.App()
app.add_page(index)
