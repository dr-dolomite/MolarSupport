import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { FaTrashCan } from "react-icons/fa6";
import { TiWarningOutline } from "react-icons/ti";


interface DeleteIconProps {
    onClick: () => void;
    disabled?: boolean;
  }
  
  export const DeleteIcon = ({ onClick, disabled }: DeleteIconProps) => {
    return (
      <Button variant="outline" size="lg" onClick={onClick} disabled={disabled}>
        <FaTrashCan className="text-destructive size-6" />
      </Button>
    );
  };