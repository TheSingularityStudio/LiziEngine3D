use clap::{Parser, Subcommand};
use lizi_engine_3d::gui::LiziApp;

#[derive(Parser)]
#[command(name = "lizi3d", about = "3D Electrostatic PIC Simulator")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    Gui {
        #[arg(long, default_value = "1000")]
        width: u32,
        #[arg(long, default_value = "800")]
        height: u32,
    },
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        None | Some(Command::Gui { .. }) => {
            LiziApp::run();
        }
    }
}