import { useState, useEffect } from 'react'
import { Card, Table, Button, Modal, Form, Input, message } from 'antd'
import { unitsApi } from '../../api'

export default function SettingsUnits() {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [modalOpen, setModalOpen] = useState(false)
  const [editingId, setEditingId] = useState(null)
  const [form] = Form.useForm()

  const load = async () => {
    setLoading(true)
    try {
      const res = await unitsApi.list()
      setData((res.data?.data || []).map((u, i) => ({ ...u, key: u.id || i })))
    } catch {
      message.error('加载失败')
      setData([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const handleAdd = () => {
    setEditingId(null)
    form.resetFields()
    setModalOpen(true)
  }

  const handleEdit = (record) => {
    setEditingId(record.id)
    form.setFieldsValue({ name: record.name, code: record.code, region: record.region })
    setModalOpen(true)
  }

  const handleDelete = (record) => {
    Modal.confirm({
      title: '确认删除',
      content: `确定删除单位「${record.name}」吗？`,
      okText: '确定',
      cancelText: '取消',
      onOk: async () => {
        try {
          await unitsApi.delete(record.id)
          message.success('已删除')
          load()
        } catch (e) {
          message.error(e.response?.data?.detail || '删除失败')
        }
      },
    })
  }

  const handleSubmit = async () => {
    try {
      const v = await form.validateFields()
      if (editingId) {
        await unitsApi.update(editingId, v)
        message.success('已更新')
      } else {
        await unitsApi.create(v)
        message.success('已添加')
      }
      setModalOpen(false)
      load()
    } catch (e) {
      if (e.errorFields) return
      message.error(e.response?.data?.detail || '操作失败')
    }
  }

  return (
    <Card className="glass-card" title="单位管理">
      <Button type="primary" className="tech-btn" style={{ marginBottom: 16 }} onClick={handleAdd}>新增单位</Button>
      <Table
        dataSource={data}
        loading={loading}
        columns={[
          { title: '单位名称', dataIndex: 'name' },
          { title: '编号', dataIndex: 'code' },
          { title: '所属区域', dataIndex: 'region' },
          {
            title: '操作',
            render: (_, r) => (
              <>
                <Button type="link" size="small" onClick={() => handleEdit(r)}>编辑</Button>
                <Button type="link" size="small" danger onClick={() => handleDelete(r)}>删除</Button>
              </>
            ),
          },
        ]}
      />
      <Modal
        title={editingId ? '编辑单位' : '新增单位'}
        open={modalOpen}
        onOk={handleSubmit}
        onCancel={() => setModalOpen(false)}
        okText="确定"
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="单位名称" rules={[{ required: true }]}>
            <Input placeholder="如：彭水县供电区域" />
          </Form.Item>
          <Form.Item name="code" label="编号" rules={[{ required: true }]}>
            <Input placeholder="如：U001" />
          </Form.Item>
          <Form.Item name="region" label="所属区域">
            <Input placeholder="如：重庆市彭水县" />
          </Form.Item>
        </Form>
      </Modal>
    </Card>
  )
}
