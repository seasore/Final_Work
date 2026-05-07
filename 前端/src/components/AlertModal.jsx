import { Modal } from 'antd'
import { WarningOutlined } from '@ant-design/icons'

/** 告警弹窗 - 玻璃质感，准确率低于阈值或负荷偏差超范围时触发 */
export default function AlertModal({ open, onClose, title = '系统告警', content, type = 'warning' }) {
  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={420}
      styles={{
        content: {
          background: 'rgba(15, 30, 50, 0.95)',
          backdropFilter: 'blur(20px)',
          border: '1px solid rgba(255, 100, 100, 0.5)',
          boxShadow: '0 0 30px rgba(255, 100, 100, 0.2)',
        },
      }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16 }}>
        <WarningOutlined style={{ fontSize: 32, color: '#ff4d4f' }} />
        <div>
          <h3 style={{ color: '#e8f4ff', marginBottom: 8 }}>{title}</h3>
          <p style={{ color: 'rgba(255,255,255,0.8)' }}>{content}</p>
        </div>
      </div>
    </Modal>
  )
}
