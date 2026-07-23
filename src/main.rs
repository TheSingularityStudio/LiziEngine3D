use clap::{Parser, Subcommand};

use lizi_engine_2d::gui::LiziApp;

#[derive(Parser)]
#[command(name = "lizi2d", about = "2D Electrostatic PIC Simulator")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// 启动 GUI 窗口（默认模式）
    Gui {
        #[arg(long, default_value = "800")]
        width: u32,
        #[arg(long, default_value = "700")]
        height: u32,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        None | Some(Command::Gui { .. }) => {
            // 默认启动 GUI
            LiziApp::run();
        }
    }
}